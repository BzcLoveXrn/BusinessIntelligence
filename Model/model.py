from datetime import datetime

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# 定义标签列表
LABELS = ['入门', '高端', '性价比', '暴力', '进攻', '杀球', '控制',
          '头重', '连贯', '速度', '中杆硬', '中杆软', '糖水', '颜值', '拉吊']


class BadmintonDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.FloatTensor(self.labels[idx])

        return item


class BadmintonBertClassifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return self.sigmoid(logits)




def save_model(model, tokenizer, save_dir='models'):
    """保存模型和分词器"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(save_dir, 'badminton_model.pt')
    torch.save(model.state_dict(), model_path)

    # 保存分词器
    tokenizer_path = os.path.join(save_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)

    print(f"模型已保存到: {save_dir}")


def load_model(model_dir='models'):
    """加载保存的模型和分词器"""
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))

    # 初始化模型
    model = BadmintonBertClassifier(len(LABELS))

    # 加载模型权重
    model_path = os.path.join(model_dir, 'badminton_model.pt')
    model.load_state_dict(torch.load(model_path))

    return model, tokenizer


def predict(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = (outputs > 0.5).int().cpu().numpy()

    predicted_labels = [LABELS[i] for i in range(len(LABELS)) if predictions[0][i] == 1]
    return predicted_labels


class Tools:
    def __init__(self, path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_model, self.loaded_tokenizer = load_model(path)

    def get_predict_values(self, comment):
        if not isinstance(comment, str):
            comment = ""  # 或者使用 "" 来处理空值
        predicted_labels = predict(comment, self.loaded_model, self.loaded_tokenizer, self.device)
        return predicted_labels


# 示例使用代码
def train_model(model, train_loader, criterion, optimizer, device, epoch, save_dir='checkpoints'):
    model.train()
    total_loss = 0
    batch_losses = []  # Track loss for each batch

    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())

        if (batch_idx + 1) % 100 == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, f'{save_dir}/checkpoint_batch_{batch_idx}.pt')

    return total_loss / len(train_loader), batch_losses


def plot_training_metrics(epoch_losses, batch_losses_history, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot epoch losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{save_dir}/epoch_loss_{timestamp}.png')
    plt.close()

    # Plot batch losses
    plt.figure(figsize=(12, 6))
    for epoch, batch_losses in enumerate(batch_losses_history):
        plt.plot(batch_losses, label=f'Epoch {epoch + 1}', alpha=0.7)
    plt.title('Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/batch_loss_{timestamp}.png')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BadmintonBertClassifier(len(LABELS)).to(device)
    train_data = pd.read_csv("../DataAnalysis/data/train_data.csv")

    train_texts = train_data['comment'].tolist()
    train_labels = []

    for _, row in train_data.iterrows():
        label_row = [1 if row[label] != 0 else 0 for label in LABELS]
        train_labels.append(label_row)

    train_dataset = BadmintonDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    num_epochs = 15
    best_loss = float('inf')
    epoch_losses = []
    batch_losses_history = []

    print("开始训练模型")

    for epoch in range(num_epochs):
        train_loss, batch_losses = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        epoch_losses.append(train_loss)
        batch_losses_history.append(batch_losses)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, tokenizer)

        # 每个epoch结束后绘制损失图
        plot_training_metrics(epoch_losses, batch_losses_history)

    save_model(model, tokenizer, save_dir='models/final')


if __name__ == "__main__":
    main()
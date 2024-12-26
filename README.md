## 智能羽毛球推荐系统
### Quickstart
1. pip install -r requirements.txt 安装相关依赖的库
2. cd badmintoncn 进入到爬虫目录，
3. scrapy crwal [爬虫名字]    爬取数据到/DataAnalysis/data/comment_data.csv,   /DataAnalysis/data/info_data.csv,  /DataAnalysis/data/train_data.csv,   爬虫的名字为comment_spider，train_data_spider，basic_info_spider

-----------------  以上是爬取数据部分，若数据已经存在，则无需运行

5. cd Model  进入模型目录
6. python model.py 训练模型（此步之前，train_data.csv必须存在）

   
--------------------   以上是训练模型部分，若模型文件已经存在，则无需运行


8. jupyter notebook 接下来步骤在show.ipyb运行
9. 运行show.ipyb中的所有cell，在这里面会运行生成comment_data_tags.csv,sum_data.csv，其中有个异步多进程产生data.csv的步骤是预测评论，很耗时间大约2-3小时跑完

    
--------------------   以上是生成关键数据data.csv和数据可视化的过程，如data.csv已经存在，则无需运行

11. cd Web  进入web目录
12. python app.py  启动网页
 

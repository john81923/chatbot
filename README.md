# chatbot
## Model description
* Model的參數設定:
  * 使用 attention
  * layer_number = 2
  * Layer size = 512
  * Bucket size. = [(5, 10), (10, 15), (20, 25), (40, 50)]
  * Batch_size = 64
  * Learning rate = 0.001 往下調降
  * Dictionary size = 6781 以字為單位
## Improvement
* 使用attention base model
* 這次的task遇到的第一個問題就是句子的長對不一，使得很短的句子也要做很多的
padding使得memory用量很大而且，training速度會變慢。所以我們用四種不同大
小的bucket分別代表不同的input與output長度，並且適當的分配對話進入bucket中，
解決這個問題，我們只需要建立四種對應大小的training graph。
Model 部分我們用 s2s+ attention，大概train了半天，loss 落在 2.4左右
Inference 的時候，用的是schedule sampling ，隨機挑選sample

## Experimental results and settings
### input
<img height="200" src="https://github.com/john81923/chatbot/blob/master/input.png">

### output
<img height="200" src="https://github.com/john81923/chatbot/blob/master/output.png">

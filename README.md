
 # Text-Classification    

작성중...    
    
딥러닝을 활용한 문장 분류    

 ## How to Using    
 
 - **GitHub Cloning**  
```bash 
>> git clone https://github.com/dlcjfgmlnasa/Text-Classification.git --recursive
```

 - **Installing Python Package (with python virtualenv)**  
 
```bash 
>> python -m venv venv                          # create python virtualenv  
>> source venv/source/activte                   # activate virtualenv  
>> (venv) pip install -r requirements.txt       # install...  
```  
  
- **Prepare Dataset**  
  Your dataset should look like this  
  + `id`: id   
  + `document`: The actual review   
  + `label`: The sentiment class of the review. (0: negative, 1: positive)    
  + dataset line split `\t`  
   + **example**  
  <br>  
      
   |id | document | label|   
   |:-:|:--------:|:----:|  
   | 1 | 아 더빙.. 진짜 짜증나네요 목소리 | 0 |  
   | 2 | 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나 | 1 |  
   | 3 | 너무재밓었다그래서보는것을추천한다 | 1 |  
   | 4 | 교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정 | 0 |  
   | 5 | 막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움. | 0 |  
   | 6 | 원작의 긴장감을 제대로 살려내지못했다. | 0 |  
   | 7 | 액션이 없는데도 재미 있는 몇안되는 영화 | 1 |  
   | 8 | 재미없다 지루하고. 같은 음식 영화인데도 바베트의 만찬하고 넘 차이남....바베트의 만찬은 이야기도 있고 음식 보는재미도 있는데 ; 이건 볼게없다 음식도 별로 안나오고, 핀란드 풍경이라도 구경할랫는데 그것도 별로 안나옴 | 0 |  
   |...| ... | ... |  
  
- **Training**  
```python    
 ``` 

- **Predicate**  
```python    
 ```    
 
 ## Requirements  
 
- Python 3.6 (may work with other versions, but I used 3.6)    
- PyTorch 1.2.0    
- konlpy 0.5.1    
    
## Datasets   

- Naver sentiment movie corpus v1.0 사용 
- https://github.com/e9t/nsmc     
    
## Model 

### 목차    

1. [TextCNN](####-1.TextCNN)
2. [TextRNN](####-2.TextRNN)
3. [BiLSTM with Attention](####-3.BiLSTM_with_Attention)
4. [Self Attention]()
    
#### 1. TextCNN  

> **paramter**

<center>

| epoch | batch_size | seq_len | embedding_dim | output_channels | dropout_rate | n_grams |
|:-----:|:----------:|:-------:|:-------------:|:---------------:|:------------:|:-------:|
|  20   |    500     |    20   |      512      |       50	 |      0.8	| [2,3,4] |

</center>

> **Training Graph**

![TextCNN Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/text_cnn_accuracy_loss.PNG?raw=true)

> **Test**

![TestCNN Test Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/text_cnn_test_result.PNG?raw=true)

#### 2. TextRNN

> **paramter**

<center>

| epoch | batch_size | seq_len | embedding_dim | rnn_dim | rnn_num_layer | bidirectional |
|:-----:|:----------:|:-------:|:-------------:|:-------:|:-------------:|:-------------:|
|  20   |    500     |    20   |      512      |   50	 |       2       |      True     |

</center>

> **Training Graph**
 
![TextRNN Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/text_rnn_accuracy_loss.PNG?raw=true)

> **Test**

![TextRNN Test Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/text_rnn_test_result.PNG?raw=true)

#### 3. BiLSTM with Attention

> **paramter**

<center>

| epoch | batch_size | seq_len | embedding_dim | rnn_dim | rnn_num_layer | bidirectional |
|:-----:|:----------:|:-------:|:-------------:|:-------:|:-------------:|:-------------:|
|  20   |    500     |    20   |      512      |   50	 |       2       |      True     |

</center>

> **Training Graph**

![BiLSTM Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/bi_rnn_with_attention_accuracy_loss.PNG?raw=true)

> **Test**

![BiLSTM Test Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/bi_rnn_with_attention_test_result.PNG?raw=true)

#### 4. Self Attention

> **paramter**

<center>

| epoch | batch_size | seq_len | embedding_dim | self_attention_dim | self_attention_num_heads |
|:-----:|:----------:|:-------:|:-------------:|:---------:|:---------:|
|  20   |    500     |    20   |      512      |   64	     |     8     |

</center>

> **Training Graph**

![Self Attention Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/self_attention_accuracy_loss.PNG?raw=true)

> **Test**

![Self Attention Test Result Image](https://github.com/dlcjfgmlnasa/Text-Classification/blob/master/image/self_attention_test_result.PNG?raw=true)
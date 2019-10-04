
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
    + example)
	```bash
	id	document	label  
	1	아 더빙.. 진짜 짜증나네요 목소리	0
	2	흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나	1
	3	너무재밓었다그래서보는것을추천한다	1
	4	교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정	0
	5	막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.	0
	6	원작의 긴장감을 제대로 살려내지못했다.	0
	7	액션이 없는데도 재미 있는 몇안되는 영화	1
	....
	```

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
  
1. [TextCNN](#1.-TextCNN)  
2. [TextRNN](#2.-TextRNN)  
  
### 1. TextCNN  
  
### 2. TextRNN

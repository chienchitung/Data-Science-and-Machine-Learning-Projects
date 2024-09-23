
## 研究主題: Twitter情感分析預測Netflix股價波動

## 研究背景

1. 研究動機
   - 預測股價波動對投資者、金融分析師來說，是一個複雜的挑戰，因為股市在全球經濟扮演至關重要的角色。金融市場的時間序列經常具有高雜訊及非線性等特徵，因此提高預測準確率是非常具有挑戰性的任務。
   - 大多過去研究會透過Yahoo Finance的歷史股市數據，藉由不同的機器學習理論模型來預測股市，但有研究指出，這類型的研究多半帶有不確定性及缺陷。過去幾年，由於社群媒體普及，一些研究將社群媒體的情感數據納入模型中，來預測股市。
   - 本研究以 Abdelfattah et al. (2024) 作為 key paper，參考其建模架構，並選擇Netflix為股票標的，利用Twitter上有關Netflix的推文，使用情感分類套件(TextBlob、VADER)進行情感分析和情感分類，判斷文本中是正面、負面還是中性，並結合股市數據，透過機器學習建立預測股票波動模型。

2. 研究問題
   - 透過不同情感分析模型分類結果，結合機器學習來準確預測股票波動
   - RQ1: 比較不同種機器學習模型，有無加入情感數據，模型預測效能是否產生顯著影響
   - RQ2: 比較不同種機器學習模型，加入TextBlob跟VADER情感分析後，模型預測效能是否有差異

3. 模型實驗設計
   - 股價預測漲跌
   - 股價 + TextBlob預測漲跌
   - 股價 + Vader預測漲跌

4. 預期目標
   - 本研究目標是提供一個結合情感分析的預測模型，來更加準確預測股票波動。
   - 評估不同機器學習模型(LSTM、SVM、Random Forset、Prophet)，比較預測模型之間的準確度，提供未來研究建議。

## 研究方法
### 研究架構
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/project_%20structure.PNG)</center>

### 機器學習建模流程
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/building_model.PNG)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/research_structure.PNG)</center>

### 資料前處理
1. 讀取Twitter的貼文後，並進行貼文資料的前處理，將語意無關的符號或字串將予以刪除。
2. 使用TextBlob和Vader情感套件，分別計算出Polarity及Compound的情感分數。
3. 匯入Yahoo Finance的資料後，將情感資料與股價資料做整併。
4. 為了將模型有更好的訓練，因此加入其他技術指標及前五日收盤價作為特徵變數到資料集。
5. 將當日收盤價減去前一日收盤價進行漲跌二分類的標記，做為目標變數。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/textblob_vader_info.PNG)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/dataset_with_close_lag.PNG)</center>

### 探索性分析
1. 讀取

## 預測結果

## 參考資料
1. 林群崴，《訊號分解對於長短期記憶預測股價準確率之影響》，碩士論文，國立臺灣大學電機資訊學院資訊工程學系，2020。
2. 陳俊達，《以文件分類技術預測股價趨勢》，碩士論文，國立政治大學資訊科學系，2007。
3. Abdelfattah BA, Darwish SM, Elkaffas SM. Enhancing the Prediction of Stock Market Movement Using Neutrosophic-Logic-Based Sentiment Analysis. Journal of Theoretical and Applied Electronic Commerce Research. 2024; 19(1):116-134
4. Cristescu MP, Nerisanu RA, Mara DA, Oprea S-V. Using Market News Sentiment Analysis for Stock Market Prediction. Mathematics. 2022; 10(22):4255.
5. Koukaras P, Nousi C, Tjortjis C. Stock Market Prediction Using Microblogging Sentiment Analysis and Machine Learning. Telecom. 2022; 3(2):358-378.
6. Asghar, M.Z.; Rahman, F.; Kundi, F.M.; Ahmad, S. Development of stock market trend prediction system using multiple regression. Computational and Mathematical Organization Theory. 2019, 25, 271–301.
7. Mubeena SK, Kumar MA, Ramya U, Sujatha P. Forecasting Stock Market Movement Direction Using Sentiment Analysis and Support Vector Machine. International Research Journal of Engineering and Technology. 2020;17(3):803-808.
8. Daori H, Alharthi M, Alanazi A, et al. Predicting Stock Prices Using the Random Forest Classifier. [Published November 14, 2022].








   


## 研究主題: Twitter情感分析預測Netflix股價波動

## 研究背景

- 研究動機
   - 預測股價波動對投資者、金融分析師來說，是一個複雜的挑戰，因為股市在全球經濟扮演至關重要的角色。金融市場的時間序列經常具有高雜訊及非線性等特徵，因此提高預測準確率是非常具有挑戰性的任務。
   - 大多過去研究會透過Yahoo Finance的歷史股市數據，藉由不同的機器學習理論模型來預測股市，但有研究指出，這類型的研究多半帶有不確定性及缺陷。過去幾年，由於社群媒體普及，一些研究將社群媒體的情感數據納入模型中，來預測股市。
   - 本研究以 Abdelfattah et al. (2024) 作為 key paper，參考其建模架構，並選擇Netflix為股票標的，利用Twitter上有關Netflix的推文，使用情感分類套件(TextBlob、VADER)進行情感分析和情感分類，判斷文本中是正面、負面還是中性，並結合股市數據，透過機器學習建立預測股票波動模型。

- 研究問題
   - 透過不同情感分析模型分類結果，結合機器學習來準確預測股票波動
   - RQ1: 比較不同種機器學習模型，有無加入情感數據，模型預測效能是否產生顯著影響
   - RQ2: 比較不同種機器學習模型，加入TextBlob跟VADER情感分析後，模型預測效能是否有差異

- 模型實驗設計
   - 股價預測漲跌
   - 股價 + TextBlob預測漲跌
   - 股價 + Vader預測漲跌

- 預期目標
   - 本研究目標是提供一個結合情感分析的預測模型，來更加準確預測股票波動。
   - 評估不同機器學習模型(LSTM、SVM、Random Forset、Prophet)，比較預測模型之間的準確度，提供未來研究建議。

## 研究方法
### 研究架構
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/project_%20structure.png)</center>

### 機器學習建模流程
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/building_model.png)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/research_flow.png)</center>

### 資料前處理
這次的資料來源是從Kaggle下載的Twitter數據，原始資料的日期範圍從2020年1月至2022年7月，共涵蓋635天，包含36萬則貼文，這些資料將用於模型訓練。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/netflix_dataset.png)</center>

1. 讀取Twitter的貼文後，並進行貼文資料的前處理，將語意無關的符號或字串將予以刪除。
2. 使用TextBlob和Vader情感套件，分別計算出Polarity及Compound的情感分數。
3. 匯入Yahoo Finance的資料後，將情感資料與股價資料做整併。
4. 為了將模型有更好的訓練，因此加入其他技術指標及前五日收盤價作為特徵變數到資料集。
5. 將當日收盤價減去前一日收盤價進行漲跌二分類的標記，做為目標變數。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/textblob_vader_info.png)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/dataset_with_close_lag.png)</center>

### 探索性分析
- 在讀取並清理完資料後，首先觀察Netflix股票收盤價的趨勢變化，以確認是否存在任何異常情況。數據顯示共有635個交易日，整體趨勢呈現先漲後跌的走勢。在圖表下方，綠色點表示當天股價上漲，紅色點則代表下跌。分析結果顯示，樣本中漲跌次數大致均衡，各佔約50%。從分布情況來看，並未發現明顯的不均衡現象。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/close_price_trend.png)</center>

- 對貼文情緒分類進行資料占比分析後發現，TextBlob分析結果中，中性貼文佔最多，達到43.5%。相比之下，VADER的分析結果顯示更多情緒化的傾向，正面評論增加至42.4%，而負面評論則增加至17.4%。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/sentiment_analysis.png)</center>

## 模型建立
預測結果將透過混淆矩陣以及四個常用於評估分類模型的指標來展示各組資料和模型的表現。混淆矩陣中，行表示實際的漲跌情況，列表示預測的漲跌情況。綠色區域代表預測與實際一致的部分，顏色越深或比例越大表示預測越準確。四個評估指標包括準確率、精確率、召回率和F1分數。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/confusion_matrix.png)</center>

### LSTM
- 最左邊顯示的是僅使用價格資料的預測結果，準確率約為60%。主要問題出現在混淆矩陣的右上角，實際為跌的樣本容易被誤預測為漲。
- 加入TextBlob後，實際為跌的樣本被誤預測為漲的情況有所改善，但同時右下角實際為漲且被正確預測為漲的比率也下降，因此整體準確率沒有顯著變化。
- 加入VADER後，實際為跌的樣本被誤預測為漲的情況並未改善，但實際為漲且被正確預測為漲的比率有所增加，因此整體準確率、精確率和F1分數均有提升。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/LSTM.png)</center>  
   
### SVM
- 在結合TextBlob情感分析資料時，線性核函數的預測準確率顯著提升，顯示SVM對於情感資料的敏感性。這表明TextBlob所提供的情感特徵能夠有效地幫助線性SVM模型改善預測能力。
- 其他核函數（如Sigmoid、Polynomial、RBF）在加入情感分析資料後，預測準確率改善有限，這可能是因為這些核函數在處理高維度非線性特徵時，對情感資料的增益不如線性核明顯。
- 若觀察Recall及F1 score的表現，不論是否加入情感分析資料，高斯核函數都是優於其他核函數。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/SVM_1.png)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/SVM_2.png)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/SVM_3.png)</center>  

### Random forest
- Random Forest Classifier在加入情感分析資料後，能夠略微提升對股市漲跌的預測準確率，但增益不如其他模型顯著。
- Random Forest Regressor在處理股價資料時表現良好，但情感分析資料對其預測能力的增益有限。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/rf_classifier.png)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/rf_regression.png)</center>  

### Prophet
- Prophet模型在處理股價資料時，能夠有效地捕捉長期趨勢和季節性變化。
- 加入情感分析資料（TextBlob或VADER）後，對於Prophet的增益有限，特別是在情感特徵未能顯著影響長期趨勢的情況下。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/prophet_1.png)</center>
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/prophet_2.png)</center>

## 評估模型
- Prophet模型在處理股價資料時的預測準確率相對穩定，尤其在長期趨勢和季節性變化的捕捉上表現良好。當加入情感分析資料（TextBlob或VADER）後，Prophet的準確率變化不大，這可能是因為該模型主要依賴於時間序列資料。
- Random Forest Classifier在處理結構化數據（如單純的股價資料）時，通常能夠提供較高的準確率。當加入情感分析資料時，準確率可能會略有提升，但增益有限。
- 在預測股價變動幅度上，Random Forest Regressor通常能夠提供良好的準確率，特別是在處理非線性和高維度數據時。加入情感分析資料後，準確率的提升取決於情感特徵是否能夠提供額外的有效資訊。
<br><center> ![image](/ML-Twitter情緒分析預測Netflix股價波動/images/final_evaluation.png)</center>

## 結論建議
### 模型選擇
- LSTM：加入VADER情感分析資料後，預測準確率提升，顯示對股市漲跌預測有幫助。這表明LSTM在處理時間序列數據時，能有效利用情感分析資料來增強預測能力。
- SVM：加入TextBlob情感分析資料後，線性模型較其他三種模型的預測準確率有顯著提升，顯示對股市漲跌預測有幫助。這顯示了SVM對於情感數據的敏感性，特別是在線性核函數下。
- Random Forest：加入情感分析資料後，預測準確率變化不大，顯示對股市漲跌預測無顯著幫助。這可能是因為隨機森林模型在處理高維度數據時，對情感資料的增益不如其他模型明顯。
- Prophet：加入情感分析資料後，預測準確率變化不大，時間序列模型在本次實驗中相對更能預測出股市趨勢。這表明Prophet在時間序列預測中的穩定性，但對情感資料的敏感度有限。
- 總體而言，加入情感分析數據（特別是VADER和TextBlob）可以在某些情況下提升預測模型的準確率，但不同模型對於情感數據的敏感度有所不同，需根據具體情況選擇合適的模型和數據組合。

### 模型應用
- 若希望通過情感分析數據提升預測準確率，建議優先考慮使用LSTM和SVM。這兩種模型在結合情感數據後，能夠提供更好的預測性能。
- 若更注重模型穩定性和準確率，RandomForest Regressor和Prophet在單純使用股價數據時表現最佳，適合用於不依賴情感資料的長期趨勢預測。

### 未來研究方向
- 探索其他情感分析工具或方法，並比較其對不同模型的影響，以尋找更優的情感資料處理方式。
- 考慮將其他類型的數據，如新聞事件、經濟指標等，納入模型，以進一步提升預測準確率。
- 深入研究情感分析資料與股價波動之間的關聯性，尤其是在不同市場條件下的表現差異。

## 參考資料
1. 林群崴，《訊號分解對於長短期記憶預測股價準確率之影響》，碩士論文，國立臺灣大學電機資訊學院資訊工程學系，2020。
2. 陳俊達，《以文件分類技術預測股價趨勢》，碩士論文，國立政治大學資訊科學系，2007。
3. Abdelfattah BA, Darwish SM, Elkaffas SM. Enhancing the Prediction of Stock Market Movement Using Neutrosophic-Logic-Based Sentiment Analysis. Journal of Theoretical and Applied Electronic Commerce Research. 2024; 19(1):116-134
4. Cristescu MP, Nerisanu RA, Mara DA, Oprea S-V. Using Market News Sentiment Analysis for Stock Market Prediction. Mathematics. 2022; 10(22):4255.
5. Koukaras P, Nousi C, Tjortjis C. Stock Market Prediction Using Microblogging Sentiment Analysis and Machine Learning. Telecom. 2022; 3(2):358-378.
6. Asghar, M.Z.; Rahman, F.; Kundi, F.M.; Ahmad, S. Development of stock market trend prediction system using multiple regression. Computational and Mathematical Organization Theory. 2019, 25, 271–301.
7. Mubeena SK, Kumar MA, Ramya U, Sujatha P. Forecasting Stock Market Movement Direction Using Sentiment Analysis and Support Vector Machine. International Research Journal of Engineering and Technology. 2020;17(3):803-808.
8. Daori H, Alharthi M, Alanazi A, et al. Predicting Stock Prices Using the Random Forest Classifier. [Published November 14, 2022].








   

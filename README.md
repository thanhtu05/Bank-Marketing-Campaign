
# Women's Clothing E-Commerce Reviews
### Customer Review Classification Using NLP & Machine Learning

## 1. Project Overview

### Mô tả bài toán

Một sàn thương mại điện tử thời trang nữ có hàng chục nghìn đánh giá sản phẩm bằng văn bản tự do, kèm điểm rating (1–5 sao) và nhãn "có giới thiệu sản phẩm hay không".

Đội sản phẩm muốn:

1. **Tự động phát hiện sớm các đánh giá KHÔNG hài lòng** chỉ từ nội dung văn bản, không cần chờ khách hàng tự đánh dấu.
2. **Hiểu bộ phận/loại sản phẩm nào đang bị đánh giá kém nhất** để ưu tiên cải thiện chất lượng sản phẩm và trải nghiệm khách hàng.

### Mục tiêu dự án

- Xây dựng hệ thống phân loại đánh giá khách hàng sử dụng Natural Language Processing (NLP).
- Dự đoán khả năng khách hàng giới thiệu sản phẩm thông qua nội dung đánh giá.
- So sánh hiệu quả của hai thuật toán **Logistic Regression** và **Decision Tree**.
- Ưu tiên phát hiện các đánh giá tiêu cực nhằm hỗ trợ doanh nghiệp cải thiện chất lượng dịch vụ.

---

## 2. Dataset

**Dataset:** Women's E-Commerce Clothing Reviews

**Nguồn:** [Kaggle - Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

Bộ dữ liệu gồm **23,486 đánh giá** sản phẩm thời trang nữ.

### Các thuộc tính chính

| Feature | Description |
|---------|-------------|
| Clothing ID | Mã sản phẩm |
| Age | Tuổi khách hàng |
| Title | Tiêu đề đánh giá |
| Review Text | Nội dung đánh giá |
| Rating | Điểm đánh giá từ 1 đến 5 |
| Recommended IND | Nhãn khách hàng có giới thiệu sản phẩm hay không |
| Positive Feedback Count | Số lượt phản hồi tích cực |
| Division Name | Nhóm sản phẩm |
| Department Name | Bộ phận sản phẩm |
| Class Name | Loại sản phẩm |

### Target Variable

`Recommended IND` là biến mục tiêu của bài toán.

| Label | Meaning |
|-------|---------|
| 0 | Not Recommended – Không giới thiệu |
| 1 | Recommended – Có giới thiệu |

**Lưu ý:** Lớp 0 là nhóm đặc biệt quan trọng vì đại diện cho những khách hàng không giới thiệu sản phẩm.

---

## 3. Technologies & Libraries

**Programming Language:** Python

**Data Processing:**
- Pandas
- NumPy
- NLTK

**Machine Learning:**
- Scikit-learn
- Logistic Regression
- Decision Tree
- RandomOverSampler
- TF-IDF Vectorizer

**Data Visualization:**
- Matplotlib
- Seaborn

**Development Environment:**
- Google Colab
- Jupyter Notebook

---

## 4. Machine Learning Workflow

### Step 1: Data Understanding

Khám phá và kiểm tra dữ liệu ban đầu:

- Kiểm tra kích thước dữ liệu bằng `shape`.
- Xem các bản ghi đầu tiên bằng `head()`.
- Kiểm tra kiểu dữ liệu bằng `info()`.
- Kiểm tra các giá trị thiếu (Missing Values).
- Phân tích phân bố của biến mục tiêu `Recommended IND`.

### Step 2: Data Preprocessing

Thực hiện tiền xử lý dữ liệu văn bản:

1. **Handling Missing Values:** Xử lý các giá trị thiếu trong `Title` và `Review Text`.
2. **Lowercasing:** Chuyển toàn bộ văn bản thành chữ thường.
3. **Removing Special Characters:** Loại bỏ các ký tự đặc biệt không cần thiết.
4. **Tokenization:** Tách văn bản thành các từ riêng lẻ.
5. **Stopword Removal:** Loại bỏ các từ ít mang ý nghĩa phân loại.

Ví dụ:

**Original Review:**
> This dress is absolutely beautiful! I really love the quality.

**After Preprocessing:**
> dress absolutely beautiful really love quality

### Step 3: Feature Extraction – TF-IDF

Sử dụng **Term Frequency – Inverse Document Frequency (TF-IDF)** để chuyển đổi văn bản thành vector số.

TF-IDF giúp xác định mức độ quan trọng của từ trong một đánh giá so với toàn bộ tập văn bản.

Công thức:

$$
TFIDF(t,d)=TF(t,d)\times IDF(t)
$$

Trong đó:

- **TF:** Tần suất xuất hiện của từ trong văn bản.
- **IDF:** Mức độ hiếm của từ trong toàn bộ tập dữ liệu.

TF-IDF phù hợp với bài toán phân loại văn bản do có khả năng biểu diễn dữ liệu văn bản dưới dạng vector thưa (Sparse Matrix).

### Step 4: Handling Imbalanced Data

Bộ dữ liệu có sự mất cân bằng giữa hai lớp:

- **Class 1:** Khách hàng giới thiệu sản phẩm – chiếm đa số.
- **Class 0:** Khách hàng không giới thiệu sản phẩm – chiếm thiểu số.

Sử dụng **RandomOverSampler** để tăng số lượng mẫu của lớp thiểu số trong tập huấn luyện.

Việc Oversampling chỉ được thực hiện trên tập Train nhằm tránh Data Leakage.

### Step 5: Model Training

Huấn luyện và so sánh hai thuật toán:

#### Model 1: Logistic Regression

Logistic Regression là thuật toán phân loại tuyến tính, phù hợp với bài toán phân loại nhị phân.

**Ưu điểm:**
- Phù hợp với dữ liệu TF-IDF nhiều chiều.
- Tốc độ huấn luyện nhanh.
- Có khả năng tổng quát hóa tốt.
- Dễ triển khai và giải thích kết quả.

#### Model 2: Decision Tree

Decision Tree dự đoán dựa trên các quy tắc phân chia dữ liệu.

**Ưu điểm:**
- Dễ trực quan hóa.
- Có khả năng mô hình hóa mối quan hệ phi tuyến.
- Dễ giải thích các quyết định phân loại.

**Hạn chế:**
- Có nguy cơ Overfitting.
- Có thể hoạt động kém hiệu quả trên dữ liệu văn bản có số lượng đặc trưng lớn.

---

## 5. Model Evaluation

Để đánh giá hiệu quả mô hình, dự án sử dụng:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

### Experimental Results

| Metric | Logistic Regression | Decision Tree |
|--------|---------------------|---------------|
| Accuracy | **89%** | 63% |
| Precision (Class 0) | **65%** | 30% |
| Recall (Class 0) | **84%** | 83% |
| F1-score (Class 0) | **73%** | 44% |

### Model Comparison

Kết quả thực nghiệm cho thấy **Logistic Regression đạt hiệu quả tổng thể tốt hơn Decision Tree**.

Đặc biệt:

- Logistic Regression đạt Accuracy khoảng 89%.
- Precision lớp 0 đạt 65%.
- Recall lớp 0 đạt 84%.
- F1-score lớp 0 đạt 73%.

Decision Tree có Recall lớp 0 tương đối cao nhưng Precision thấp, dẫn đến nhiều đánh giá tích cực bị phân loại nhầm thành không giới thiệu sản phẩm.

**Selected Model: Logistic Regression**

---

## 6. Why Is Recall for Class 0 Important?

### 6.1. Recall là gì?

Recall thể hiện khả năng nhận diện đúng các mẫu thực sự thuộc một lớp.

Đối với lớp 0:

$$
Recall_0 = \frac{TP_0}{TP_0 + FN_0}
$$

Trong đó:

- **TP₀:** Số đánh giá thực sự thuộc lớp 0 được dự đoán đúng là lớp 0.
- **FN₀:** Số đánh giá thực sự thuộc lớp 0 nhưng bị dự đoán nhầm thành lớp 1.

### 6.2. Tại sao Recall lớp 0 quan trọng?

Mục tiêu chính của dự án là phát hiện sớm những khách hàng không hài lòng.

Nếu mô hình dự đoán nhầm một đánh giá không hài lòng thành đánh giá tích cực, doanh nghiệp có thể bỏ lỡ một phản hồi quan trọng.

Điều này có thể dẫn đến:

- Không phát hiện kịp thời vấn đề chất lượng sản phẩm.
- Bỏ sót những khách hàng đang có trải nghiệm tiêu cực.
- Chậm xử lý các khiếu nại tiềm ẩn.
- Giảm khả năng cải thiện sản phẩm và dịch vụ.
- Gia tăng nguy cơ mất khách hàng.

### 6.3. Ví dụ thực tế

Giả sử có **100 đánh giá thực sự thuộc lớp 0**.

Với Recall lớp 0 bằng **84%**:

- Mô hình phát hiện đúng khoảng **84 đánh giá không giới thiệu**.
- Khoảng **16 đánh giá không giới thiệu** bị bỏ sót.

Điều này cho thấy mô hình có khả năng nhận diện phần lớn các phản hồi không hài lòng.

### 6.4. Tại sao không chỉ sử dụng Accuracy?

Do dữ liệu mất cân bằng, phần lớn khách hàng thuộc nhóm giới thiệu sản phẩm.

Nếu một mô hình luôn dự đoán khách hàng sẽ giới thiệu sản phẩm, Accuracy vẫn có thể tương đối cao nhưng không phát hiện được các đánh giá thuộc lớp 0.

Vì vậy:

> **Accuracy cao không đồng nghĩa với việc mô hình phát hiện tốt các đánh giá không hài lòng.**

Trong bài toán này, Recall lớp 0 là chỉ số quan trọng để đánh giá khả năng phát hiện phản hồi tiêu cực.

Tuy nhiên, cần xem xét đồng thời **Precision và F1-score của lớp 0** để hạn chế cảnh báo sai.

---

## 7. Business Value

Dự án có thể mang lại một số giá trị cho doanh nghiệp thương mại điện tử.

### Early Detection of Negative Customer Feedback

Phát hiện sớm những đánh giá không giới thiệu sản phẩm mà không cần chờ khách hàng tự đánh dấu.

### Customer Experience Improvement

Giúp đội ngũ chăm sóc khách hàng ưu tiên xem xét các phản hồi có khả năng thể hiện sự không hài lòng.

### Product Quality Monitoring

Hỗ trợ doanh nghiệp theo dõi chất lượng sản phẩm thông qua nội dung đánh giá của khách hàng.

### Data-Driven Decision Making

Cung cấp cơ sở để doanh nghiệp phân tích phản hồi, xác định vấn đề và đề xuất các biện pháp cải thiện trải nghiệm mua sắm.

---

## 8. Project Structure

```text
Womens-Clothing-E-Commerce-Reviews/
│
├── Womens_Clothing_E_Commerce_Reviews.ipynb
│
├── Womens Clothing E-Commerce Reviews.csv
│
└── README.md
```

## 9. How to Run

### Option 1: Google Colab

1. Mở [Google Colab](https://colab.research.google.com/).
2. Upload file `Womens_Clothing_E_Commerce_Reviews.ipynb`.
3. Upload bộ dữ liệu CSV.
4. Cài đặt các thư viện còn thiếu.
5. Chạy lần lượt các cell hoặc chọn **Runtime → Run all**.

### Option 2: Jupyter Notebook

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/Womens-Clothing-E-Commerce-Reviews.git
cd Womens-Clothing-E-Commerce-Reviews
```

Cài đặt thư viện:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk imbalanced-learn jupyter
```

Khởi chạy Jupyter Notebook:

```bash
jupyter notebook
```

Mở file `.ipynb` và thực hiện các bước huấn luyện.

---

## 10. Limitations

Dự án hiện tại vẫn có một số hạn chế:

1. Mô hình truyền thống dựa trên TF-IDF chưa thể hiểu đầy đủ ngữ cảnh và sắc thái cảm xúc phức tạp.
2. Các đánh giá có chứa phủ định, mỉa mai hoặc ngôn ngữ không trực tiếp có thể bị phân loại sai.
3. Dữ liệu có sự mất cân bằng giữa hai lớp.
4. Precision lớp 0 chưa cao, vì vậy vẫn có trường hợp đánh giá tích cực bị dự đoán thành không giới thiệu.
5. Quy trình hiện tại sử dụng cả `Title` và `Review Text`; cần thử nghiệm mô hình chỉ sử dụng `Review Text` để đánh giá đúng bài toán dự đoán từ nội dung đánh giá.
6. Việc lựa chọn tham số Decision Tree cần được cải thiện bằng Validation hoặc Cross-validation thay vì dựa trên tập Test.

---

## 11. Future Improvements

Các hướng phát triển trong tương lai:

- Thử nghiệm Linear SVM, Random Forest và XGBoost.
- Sử dụng GridSearchCV để tối ưu Hyperparameters.
- Áp dụng Cross-validation để đánh giá mô hình đáng tin cậy hơn.
- Thử nghiệm Word Embeddings và BERT.
- Tối ưu Decision Threshold để cân bằng Precision và Recall lớp 0.
- Phân tích tỷ lệ đánh giá không giới thiệu theo `Department Name` và `Class Name`.
- Xây dựng dashboard trực quan hóa phản hồi khách hàng.
- Triển khai mô hình thành ứng dụng dự đoán bằng Streamlit.

---

## 12. Conclusion

Dự án xây dựng quy trình ứng dụng NLP và Machine Learning nhằm phân loại đánh giá sản phẩm thời trang nữ.

Kết quả thực nghiệm cho thấy Logistic Regression đạt hiệu quả tốt hơn Decision Tree, với Accuracy khoảng 89% và Recall lớp 0 đạt 84%.

Điều này cho thấy Logistic Regression có khả năng nhận diện phần lớn những đánh giá thuộc nhóm không giới thiệu sản phẩm.

Dự án minh họa cách ứng dụng Machine Learning để hỗ trợ doanh nghiệp thương mại điện tử phát hiện sớm phản hồi không hài lòng và hướng tới cải thiện trải nghiệm khách hàng.

---

## Author

**Hoang Thanh Tu**

- Field: Data Analytics & Machine Learning
- University: Posts and Telecommunications Institute of Technology (PTIT)
- GitHub: [github.com/thanhtu05](https://github.com/thanhtu05)

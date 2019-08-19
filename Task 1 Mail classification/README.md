### Task1: Mail classification
____
#### Giới thiệu
<ul>
<li>Mục tiêu: Phân loại email thành 2 loại (spam/ham) => Bài toán phân loại nhị phân</li>
<li>Bài toán phân loại email là bài toán học có giám sát (supervised learning) trong học máy(machine learning) vì các email đã được gán nhãn và được sử dụng để phân loại.</li>
</ul>

#### Triển khai
Để giải quyết bài toán phân loại email, hay phân loại VB nói chung, ta thực hiện theo các bước:

- Chuẩn bị dữ liệu (Dataset preparation)
- Xử lý đặc trưng dữ liệu (Feature Engineering)
- Xây dựng mô hình(Build Model)
- Tinh chỉnh mô hình và cải thiện hiệu năng(Improve Performance)


##### 1. Tiền xử lý dữ liệu (Preprocessing Data)

+ Loại bỏ các ký tự đặc biệt trong tập dữ liệu ban đầu như dấu chấm, dấu mở đóng ngoặc,... (sử dụng thư viện **gensim**)
+ Sử dụng ***Pyvi*** để tách từ Tiếng Việt, vì 1 từ Tiếng Việt có thể được tạo thành từ nhiều tiếng (vd: xe_đạp, tức_tưởi, ...)
+ Sử dụng [tập từ dừng tiếng việt](https://github.com/stopwords/vietnamese-stopwords/blob/master/vietnamese-stopwords.txt) để loại bỏ các từ dừng (vì các từ dừng xuất hiện nhiều trong VB và hầu hết trong mọi VB, nên không có tác dụng trong bài toán phân loại)

##### 2. Feature Engineering
Mục đích là chuyển các VB (email) từ dữ liệu text sang biểu diễn dưới dạng số để thực hiện phân loại
Có nhiều cách đưa dữ liệu text về dạng số
+ Count Vectors (đã thực hiện)
+ Tf-idf Vectors (đã thực hiện)
    + Word level
    + N-gram level
 + Word-embedding (chưa thực hiện trong bt này)

##### 3. Xây dựng và đánh giá mô hình(Build Model)
Sử dụng các thư viện đã có và đánh giá các mô hình phân loại trên tập dữ liệu đã có:

| Model | Valuation | Spam_test | Ham_test|
|-------|:---------:|:---------:|:-------:|
|MultinomialNB - Count Vectorizer|95.0%|100.0%|25.71%|
|BernoulliNB - Count Vectorizer|95.0%|100.0%|91.43%|
|LinearSVC - Count Vectorizer|85.0%|100.0%|91.43%|
|MultinomialNB - word_tfidf|95.0%|100.0%|31.43%|
|BernoulliNB - word_tfidf|85.0%|100.0%|91.43%|
|LinearSVC - word_tfidf|85.0%|100.0%|65.71%|

###### ***Nhận xét:*** 


#### References:
+ [Phân loại VB bằng ML - Viblo](https://viblo.asia/p/phan-loai-van-ban-tu-dong-bang-machine-learning-nhu-the-nao-phan-2-4P856PqBZY3)
+ [Algorithm ML - machinelearningcoban.com](https://machinelearningcoban.com/)
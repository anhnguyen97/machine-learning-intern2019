###Task1: Mail classification
____
####Giới thiệu
<ul>
<li>Mục tiêu: Phân loại email thành 2 loại (spam/ham) => Bài toán phân loại nhị phân</li>
<li>Bài toán phân loại email là bài toán học có giám sát (supervised learning) trong học máy(machine learning) vì các email đã được gán nhãn và được sử dụng để phân loại.</li>
</ul>

####Triển khai
Để giải quyết bài toán phân loại email, hay phân loại VB nói chung, ta thực hiện theo các bước:

- Chuẩn bị dữ liệu (Dataset preparation)
- Xử lý đặc trưng dữ liệu (Feature Engineering)
- Xây dựng mô hình(Build Model)
- Tinh chỉnh mô hình và cải thiện hiệu năng(Improve Performance)


#####1. Tiền xử lý dữ liệu (Preprocessing Data)

+ Loại bỏ các ký tự đặc biệt trong tập dữ liệu ban đầu như dấu chấm, dấu mở đóng ngoặc,... (sử dụng thư viện **gensim**)
+ Sử dụng ***Pyvi*** để tách từ Tiếng Việt, vì 1 từ Tiếng Việt có thể được tạo thành từ nhiều tiếng (vd: xe_đạp, tức_tưởi, ...)
+ Sử dụng [tập từ dừng tiếng việt](https://github.com/stopwords/vietnamese-stopwords/blob/master/vietnamese-stopwords.txt) để loại bỏ các từ dừng (vì các từ dừng xuất hiện nhiều trong VB và hầu hết trong mọi VB, nên không có tác dụng trong bài toán phân loại)

#####2. Feature Engineering
Mục đích là chuyển các VB (email) từ dữ liệu text sang biểu diễn dưới dạng số để thực hiện phân loại
Có nhiều cách đưa dữ liệu text về dạng số
+ Count Vectors (đã thực hiện)
+ Tf-idf Vectors (đã thực hiện)
    + Word level
    + N-gram level
 + Word-embedding (chưa thực hiện trong bt này)

#####3. Xây dựng và đánh giá mô hình(Build Model)
Sử dụng các thư viện đã có và đánh giá các mô hình phân loại trên tập dữ liệu đã có
#####NaivesBayes

#####Support Vector Machine(SVM)
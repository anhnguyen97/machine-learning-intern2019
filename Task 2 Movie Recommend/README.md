####Movie Recommend - Collaborative Filtering (CF)

---

1. Mô tả bài toán:

    Với một tập các phim mà một người dùng đã xem và đánh giá, hệ thống cần phải xác định (dự đoán) những bộ phim nào (chưa được xem) mà người dùng đó thích xem. 

2. Ý tưởng:

    Có 2 hướng tiếp cận:
    + user-user collaborative filtering: gợi ý các item cho user dựa trên sự tương đồng về hành vi của các user.
        
        Xác định *mức độ quan tâm* của user đối với một item(movie) dựa trên hành vi của user khác *gần giống* với user này. Sự *gần giống nhau* giữa các *user* có thể được xác định dựa trên mức độ quan tâm của các user đối với các item khác mà hệ thống đã biết. Hai người xem những bộ phim giống nhau, đánh giá các phim đó tương tự thì sẽ có khả năng cao thích cùng các bộ phim khác trong tương lai; người này sẽ thích những bộ phim mà người kia đã đánh giá cao và ngược lại. 
    
    + item-item collaborative filtering: gợi ý các item cho user dựa trên sự tương tự giữa các item.

3. Movie recommended using User-user CF:

    3.1. Xác định độ tương đồng giữa các user:
    
    Những người dùng có hành vi càng giống nhau sẽ có độ tương đồng càng lớn
    
    Đặt mức độ gần giống nhau của 2 user ui và uj là sim(ui, uj).
    
    Để đo similarity giữa các user, cần xây dựng các ***feature vector*** cho các user. Áp dụng độ đo cosine để đo độ similarity giữa các vector đó.
    #####Chuẩn hóa dữ liệu
    Dữ liệu đầu vào là Utility Matrix, gồm các user và rating của user đối với các item(movie). Do số lượng các item rất lớn trong khi mỗi user chỉ rating 1 số lượng rất nhỏ các item nên vector đặc trưng cho mỗi user sẽ bao gồm rất nhiều các giá trị None, để có thể tính toán, cần chuẩn hóa đưa các giá trị này về giá trị số thực mà không ảnh hưởng đến độ giống nhau giữa các user.
    
    Đối với các movie mà user chưa rate, đặt rating bằng rating trung bình (mean_rate) của các item mà user đó đã rated. Sau đó, trừ các rating cho mean_rate, các rating > 0 thể hiện các item mà user đó quan tâm, yêu thích và ngược lại.
      
    #####Cosine similarity
    Áp dụng độ đo cosine để tính độ similarity giữa các user theo vector đã được chuẩn hóa.
    
    3.2. Gợi ý phim cho user
    
    Để đưa ra rating của user i đối với 1 item j, ta dựa trên k neighbor user đã rating item j, *Predicted rating* được xác định là trung bình có trọng số của các rating đã được chuẩn hóa.
    
    Rating cuối cùng được cộng vs mean_rate của user i (do dữ liệu tính toán là dữ liệu đã chuẩn hóa)
    
    3.3. Đánh giá:
    
    Sủ dụng RMSE (Root Mean Square Error) để đánh giá kết quả của hệ thống
    
4. Tập dữ liệu sử dụng:

    Tập dữ liệu [Movielens](https://grouplens.org/datasets/movielens/20m/)
    
5. Tài liệu tham khảo:
    
    + [Collaborative Filtering - ML cơ bản](https://machinelearningcoban.com/2017/05/24/collaborativefiltering/)
    + [Serious Xây dựng một recommend system](https://viblo.asia/p/lam-the-nao-de-xay-dung-mot-recommender-system-rs-phan-1-aWj53V2Gl6m)
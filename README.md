# flo_cltv_prediction
MIUUL Data Science and Machine Learning Bootcamp Projesi

#### İŞ PROBLEMİ
FLO satış ve pazarlama faaliyetleri için roadmap 
belirlemek istemektedir. Şirketin orta uzun vadeli plan 
yapabilmesi için var olan müşterilerin gelecekte şirkete
sağlayacakları potansiyel değerin tahmin edilmesi
gerekmektedir.

#### Veri Seti Hikâyesi
Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) 
olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

12 Değişken, 19.945 Gözlem, 2.7MB

master_id: Eşsiz müşteri numarası <br>
order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile) <br>
last_order_channel: En son alışverişin yapıldığı kanal <br>
first_order_date: Müşterinin yaptığı ilk alışveriş tarihi <br>
last_order_date: Müşterinin yaptığı son alışveriş tarihi <br>
last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi <br>
last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi <br>
order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı <br>
order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı <br>
customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret <br>
customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret <br>
interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi <br>

# Video Walkthrough Script (5 Minutes)

## CLV Segmentation Project Demo

---

### INTRO (0:00 - 0:30)

**[Show title slide]**

"สวัสดีครับ วันนี้ผมจะพาทุกคนมาดูโปรเจกต์ Customer Lifetime Value Analysis และ Customer Segmentation

โปรเจกต์นี้ใช้ Python และ Machine Learning ในการวิเคราะห์พฤติกรรมลูกค้า พยากรณ์มูลค่าลูกค้า และแบ่งกลุ่มลูกค้าเพื่อทำ Marketing ที่ตรงเป้า

มาเริ่มกันเลย!"

---

### DATA OVERVIEW (0:30 - 1:00)

**[Show Notebook 01 - Data Exploration]**

"เราใช้ข้อมูล Online Retail Dataset ซึ่งมี transaction มากกว่า 500,000 รายการ

หลังจาก clean ข้อมูลแล้ว เราเหลือลูกค้า 4,372 ราย

**[Show key statistics]**

- Total Revenue: 8.3 ล้านปอนด์
- Average Order Value: 21 ปอนด์
- Date Range: ธันวาคม 2010 ถึง ธันวาคม 2011"

---

### COHORT ANALYSIS (1:00 - 1:45)

**[Show Notebook 02 - Cohort Analysis]**

"ต่อไปคือ Cohort Analysis เพื่อดู Retention Rate

**[Show heatmap]**

จาก Heatmap นี้ เราเห็นว่า:
- เดือนแรกหลังซื้อครั้งแรก มี 35% กลับมาซื้ออีก
- หลัง 6 เดือน เหลือแค่ 15%

นี่คือ insight สำคัญ - เราต้องโฟกัส retention ใน 30 วันแรก"

---

### RFM SEGMENTATION (1:45 - 2:30)

**[Show Notebook 03 - RFM Segmentation]**

"RFM คือ Recency, Frequency, Monetary - วิธี classic ในการแบ่งกลุ่มลูกค้า

**[Show segment pie chart]**

เราแบ่งลูกค้าออกเป็น segments:
- Champions: 18% - ลูกค้าที่ดีที่สุด
- Loyal Customers: 22% - ซื้อบ่อย
- At Risk: 15% - เคยดีแต่กำลังหาย
- Lost: 20% - ไม่ได้ซื้อนานแล้ว

**[Show revenue by segment]**

แม้ Champions มีแค่ 18% แต่สร้าง revenue 45%!"

---

### CLV MODELING (2:30 - 3:30)

**[Show Notebook 05 - CLV Modeling]**

"ตอนนี้มาถึงส่วน Machine Learning

เราทดสอบ 4 models:
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting

**[Show model comparison chart]**

Random Forest ชนะด้วย R² = 0.72

**[Show feature importance]**

Feature ที่สำคัญที่สุด:
1. Monetary - 45%
2. Frequency - 25%
3. Customer Lifetime - 12%

**[Show CLV distribution]**

Average CLV = £1,898 ต่อลูกค้า"

---

### ADVANCED SEGMENTATION (3:30 - 4:00)

**[Show Notebook 06 - Advanced Segmentation]**

"นอกจาก RFM เรายังใช้ K-Means Clustering

**[Show elbow curve and silhouette]**

Optimal clusters = 5

**[Show cluster visualization]**

แต่ละ cluster มี profile และ strategy ที่ต่างกัน"

---

### DASHBOARD DEMO (4:00 - 4:30)

**[Run Streamlit Dashboard]**

"ทุกอย่างถูก deploy บน Streamlit Dashboard

**[Navigate through pages]**

- Overview: ดู KPIs หลัก
- RFM Analysis: Interactive scatter plot
- CLV Predictions: ดู prediction ของแต่ละลูกค้า
- Segment Deep Dive: เจาะลึกแต่ละ segment

Dashboard นี้ทำให้ทีม Marketing ใช้งานได้จริง"

---

### BUSINESS IMPACT (4:30 - 5:00)

**[Show ROI slide]**

"สรุป Business Impact:

ถ้าใช้งบ Marketing £100,000 ตาม recommendations:
- Expected Return: £200,000+
- ROI: 100%+

**[Show implementation timeline]**

Implementation มี 4 phases ใช้เวลา 12 เดือน

---

**[Closing slide]**

ขอบคุณที่รับชมครับ!

โค้ดทั้งหมดอยู่บน GitHub
ถ้ามีคำถาม สามารถติดต่อได้เลยครับ

สวัสดีครับ!"

---

## Technical Notes for Recording

### Screen Recordings Needed:
1. Jupyter notebooks running (01-07)
2. Dashboard navigation
3. Charts and visualizations

### Props Needed:
- Title slide
- Key metrics slide
- ROI summary slide
- GitHub link slide

### Tips:
- Keep energy high
- Pause briefly between sections
- Highlight key numbers
- Show actual code running when possible

---

*Total runtime: 5 minutes*

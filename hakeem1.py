#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r'C:\Users\USER\Desktop\Students_Grading_Dataset.csv')
df.head()


# In[3]:


print(df.columns)


# In[4]:



print("--- معلومات عامة عن البيانات ---")
print(df.info())


# In[5]:



print("\n--- فحص القيم المفقودة (Null Values) ---")
print(df.isnull().sum())


# In[6]:



print("\n--- إحصائيات سريعة للبيانات الرقمية ---")
df.describe()


# In[7]:



drop_cols = [
    "Student_ID",
    "First_Name",
    "Last_Name",
    "Email",
    "Total_Score",
    "Final_score",
]
df_cleaned = df.drop(
    columns=[col for col in drop_cols if col in df.columns]
)


# In[8]:



df_cleaned.fillna(df_cleaned.select_dtypes(include=["number"]).mean(), inplace=True)

print(f"أبعاد البيانات بعد التنظيف: {df_cleaned.shape}")


# In[9]:


print(df_cleaned.columns)


# In[10]:



df_cleaned["Success"] = (df_cleaned["Final_Score"] >= 50).astype(int)


# In[11]:


from sklearn.model_selection import train_test_split

# تحديد المتغيرات المستخدمة للتنبؤ
selected_features = [
    "Attendance (%)",
    "Midterm_Score",
    "Study_Hours_per_Week"
]

X = df_cleaned[selected_features]
y = df_cleaned["Success"]

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("تم تقسيم البيانات بنجاح!")
print("حجم بيانات التدريب:", X_train.shape)
print("حجم بيانات الاختبار:", X_test.shape)


# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# تدريب النموذج
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# التنبؤ والتقييم
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"دقة نموذج Logistic Regression: {lr_accuracy * 100:.2f}%")


# In[13]:




from sklearn.tree import DecisionTreeClassifier

# تدريب النموذج
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# التنبؤ والتقييم
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"دقة نموذج Decision Tree: {dt_accuracy * 100:.2f}%")


# In[14]:



from sklearn.ensemble import RandomForestClassifier

# تدريب النموذج
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# التنبؤ والتقييم
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"دقة نموذج Random Forest: {rf_accuracy * 100:.2f}%")


# In[15]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# تدريب النموذج
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# التنبؤ والتقييم
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print(f"🔹 دقة نموذج KNN: {knn_acc * 100:.2f}%")


# In[16]:




from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# تدريب النموذج
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# التنبؤ والتقييم
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"🔹 دقة نموذج SVM: {svm_acc * 100:.2f}%")


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# تدريب النماذج وحساب دقتها مباشرة داخل الكود
acc_lr = accuracy_score(y_test, LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train).predict(X_test))
acc_dt = accuracy_score(y_test, DecisionTreeClassifier(random_state=42).fit(X_train, y_train).predict(X_test))
acc_rf = accuracy_score(y_test, RandomForestClassifier(random_state=42).fit(X_train, y_train).predict(X_test))
acc_knn = accuracy_score(y_test, KNeighborsClassifier().fit(X_train, y_train).predict(X_test))
acc_svm = accuracy_score(y_test, SVC(random_state=42).fit(X_train, y_train).predict(X_test))

# إنشاء جدول المقارنة وترتيبه تنازلياً
models_comparison = pd.DataFrame({
    "Model Name": ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"],
    "Accuracy (%)": [acc_lr * 100, acc_dt * 100, acc_rf * 100, acc_knn * 100, acc_svm * 100]
})

models_comparison = models_comparison.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
print(models_comparison)


# In[18]:


import gradio as gr
import numpy as np

def predict_logistic(attendance, midterm, study_hours):
    student_data = np.array([[attendance, midterm, study_hours]])
    prediction = lr_model.predict(student_data)[0]
    
    if prediction == 1:
        return "🎉 ناجح (Pass)"
    else:
        return "⚠️ راسب (Fail)"

demo = gr.Interface(
    fn=predict_logistic,
    inputs=[
        gr.Number(label="Attendance (%)"),
        gr.Number(label="Midterm Score"),
        gr.Number(label="Study Hours per Week")
    ],
    outputs=gr.Textbox(label="Logistic Regression Prediction"),
    title="نظام التنبؤ بأداء الطالب - أفضل نموذج"
)

demo.launch(inline=False, share=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





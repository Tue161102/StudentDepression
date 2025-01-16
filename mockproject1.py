import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

df= pd.read_csv (r'C:\Users\Admin\Documents\Python\StudentDepressionDataset.csv')
df.info()
print(df.head)
print(df.columns.tolist())
df.columns = df.columns.str.strip()
print(df.isnull().sum())

df.select_dtypes(include='object').nunique()

df.Gender.unique()

df[df['Gender'].isnull()].head()

df.City.unique()

def group_city(City):
    if pd.isna(City):
        return 'Other'
    
    if any(keyword in City for keyword in ['Visakhapatnam', 'Bangalore', 'Srinagar', 'Varanasi', 'Jaipur','Pune', 'Chennai']):
        return 'Northern Zone'
    elif any(keyword in City for keyword in ['Nagpur', 'Nashik', 'Vadodara','Kalyan', 'Rajkot', 'Kolkata', 'Mumbai', 'Lucknow','Indore']):
        return 'Central Zone'
    elif any(keyword in City for keyword in ['Surat', 'Ludhiana', 'Bhopal', 'Meerut', 'Agra',]):
        return 'Eastern Zone'
    elif any(keyword in City for keyword in ['Ghaziabad', 'Hyderabad', 'Vasai-Virar', 'Kanpur','Faridabad', 'Delhi', 'Saanvi','Ahmedabad','Thane']):
        return 'Western Zone'
    elif any(keyword in City for keyword in [ 'Patna', 'Rashi', 'ME', 'M.Com','Nalyan', 'Mihir', 'Nalini', 'Nandini', 'Khaziabad']):
        return 'Southern Zone'
    else:
        return 'Other'

df['city_segment'] = df['City'].apply(group_city)
df.info()
df['City'].value_counts().head(20)

df['city_segment'].value_counts().head(20)

df.Profession.unique()

def group_profession(Profession):
    if pd.isna(Profession):
        return 'Other'
    
    if any(keyword in Profession for keyword in ['Student']):
        return 'Student'
    elif any(keyword in Profession for keyword in ['Civil Engineer', 'Architect', 'UX/UI Designer']):
        return 'Engineering'
    elif any(keyword in Profession for keyword in ['Digital Marketer', 'Content Writer', 'Manager', 'Entrepreneur']):
        return 'Economics'
    elif any(keyword in Profession for keyword in ['Doctor','Pharmacist']):
        return 'Healthcare'
    elif any(keyword in Profession for keyword in [ 'Educational Consultant','Teacher']):
        return 'Education'
    elif any(keyword in Profession for keyword in ['Chef','Lawyer']):
        return 'Service'
    else:
        return 'Other'

df['profession_segment'] = df['Profession'].apply(group_profession)

df.head()

df.Degree.unique()

def degree_segment(Degree):
    if pd.isna(Degree):
        return 'Other'
    
    if any(keyword in Degree for keyword in ['B.Pharm', 'BSc', 'BA', 'BCA', 'B.Ed', 'LLB', 'BE', 'BHM', 'B.Pharm', 'B.Com', 'B.Arch', 'B.Tech']):
        return 'Undergraduate Degrees'
    elif any(keyword in Degree for keyword in ['M.Tech', 'M.Ed', 'MSc', 'M.Pharm', 'MCA', 'MA', 'MBA', 'M.Com', 'LLM', 'MHM']):
        return 'Postgraduate Degrees'
    elif any(keyword in Degree for keyword in ['PhD', 'MD', 'MBBS']):
        return 'Doctoral Degrees' 
    else:
        return 'Other'
df['degree_segment'] = df['Degree'].apply(degree_segment)

df['Degree'].value_counts().head(20)

df['degree_segment'].value_counts().head(20)

df.drop(columns = ['City', 'Profession', 'Degree','id'], inplace=True)
df.head()
df.info()

missing_percentages = (df.isnull().mean() * 100).sort_values(ascending=False)
missing_values = missing_percentages[missing_percentages > 0]
print(missing_values)
df['Financial Stress'].fillna(round(df['Financial Stress'].mean()), inplace=True)

df.rename(columns={
    'Have you ever had suicidal thoughts ?': 'SuicidalThought',
    'Family History of Mental Illness': 'FamilyMental'
    }, inplace=True)

for col in df.select_dtypes(include='object').columns:
    unique_values = df[col].unique()
    print(f"{col}: {unique_values}")
    
from sklearn.preprocessing import OrdinalEncoder

sleep = ['Others','Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
diet = ['Others','Unhealthy', 'Moderate', 'Healthy']
degree =['Other','Undergraduate Degrees', 'Postgraduate Degrees' ,'Doctoral Degrees']
ord = OrdinalEncoder(categories=[sleep, diet,degree])

columns_to_encode = ['Sleep Duration', 'Dietary Habits','degree_segment']

df[columns_to_encode] = ord.fit_transform(df[columns_to_encode])

print(df.head(8))

df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df['SuicidalThought'] = df['SuicidalThought'].map({'Yes':1,'No':0})
df['FamilyMental'] = df['FamilyMental'].map({'Yes':1,'No':0})

label_encoder = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:

    df[col] = label_encoder.fit_transform(df[col])

    print(f"{col}: {df[col].unique()}")  

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), cmap="YlGnBu", fmt='.2g', annot=True)
df.info()

X = df.drop(columns = ['Depression'], axis=1)
y = df['Depression']

x = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'max_iter': [100, 150, 200],
    'random_state': [5, 25]
}

model = LogisticRegression()
grid_search = GridSearchCV(model, grid, scoring='accuracy',cv=5)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

print(X_test)


model1 = LogisticRegression(penalty='l2',max_iter=100,random_state=5)
model1.fit(X_train,y_train)
y_pred_train = model1.predict(X_train)
print("Hệ số (Coefficients):", model1.coef_)

feature_names = X.columns  
coefficients = model1.coef_[0]  

print("Hệ số tương ứng với từng biến:\n ")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.6f}")

print("Hệ số chặn (Intercept):", model1.intercept_)

# Đánh giá trên tập huấn luyện
y_pred_train = model1.predict(X_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Tính các chỉ số
accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)

print(f"train_Accuracy: {accuracy}")
print(f"train_Precision: {precision}")
print(f"train_Recall: {recall}")
print(f"train_F1-Score: {f1}")

print('-------------------------------------')

# Dự đoán trên tập kiểm tra
y_pred = model1.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Tính các chỉ số
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"test_Accuracy: {accuracy}")
print(f"test_Precision: {precision}")
print(f"test_Recall: {recall}")
print(f"test_F1-Score: {f1}")


# Tính log-likelihood của mô hình
y_prob = model.predict_proba(X)

from sklearn.metrics import log_loss
log_likelihood = -log_loss(y, y_prob[:, 1])
print(f"Log-Likelihood của mô hình: {log_likelihood:.4f}")



import matplotlib.pyplot as plt
# Hiển thị Confusion Matrix

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, display_labels=["Không trầm cảm ", "Trầm cảm "], cmap="Blues")
plt.title("Confusion Matrix - TRAIN")
plt.show()

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,display_labels=["Không trầm cảm ", "Trầm cảm "], cmap="Blues")
plt.title("Confusion Matrix - TEST")
plt.show()


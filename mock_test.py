import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r'C:\Users\Admin\Documents\Python\StudentDepressionDataset.csv')
df.info()
df.describe().T

df.select_dtypes('object')
df.rename(columns={
    'Have you ever had suicidal thoughts ?':'SuicidalThought',
    'Family History of Mental Illness':'FamilyMental'
},inplace=True)

df['Sleep Duration'].unique()
df['Dietary Habits'].unique()


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

df['City_segment'] = df['City'].apply(group_city)

# Xem độ tập trung
sns.boxenplot(data=df,x='profession_segment',y='Age', hue='Gender')
plt.title('Biểu đồ Boxenplot cho thấy độ tuổi của người khảo sát')
plt.show()

# Có thể thấy cả nam và nữ khảo sát đều trong độ tuổi từ 20 đến 30 tuổi tuy nhiên khoảng tứ phân vị của nữ dài hơn
# Khảo sat chủ yếu tập trung vào học sinh/sinh viên tiếp đến là những ngành nghề sau khi ra trường
# Bên nam đứng thứ 2 và 3 là Engineering và Education trong khi đó nữ là Engineering và Economics 
# Sinh viên là thành phần được khảo sát nhiều nhất

bar_cols = ['Gender', 'City_segment','Sleep Duration', 'Dietary Habits','SuicidalThought', 'FamilyMental']

fig, axes = plt.subplots(3, 2, figsize=(20, 30))

axes = axes.flatten()

for ax, i in zip(axes, bar_cols):
    sns.countplot(data = df,x = i, hue='Depression',palette = 'Blues',edgecolor = 'black',ax = ax)
    ax.set_title(f'Countplot for {i}')
    for i in ax.get_xticklabels():
        i.set_rotation(45)
plt.tight_layout()
plt.show()

# tạo Ordinal Encoded
df.degree_segment.unique()
sleep = ['Others','Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
diet = ['Others','Unhealthy', 'Moderate', 'Healthy']
degree = ['Other','Undergraduate Degrees', 'Postgraduate Degrees','Doctoral Degrees']

ord = OrdinalEncoder(categories=[sleep, diet,degree])

columns_to_encode = ['Sleep Duration', 'Dietary Habits','degree_segment']

df[columns_to_encode] = ord.fit_transform(df[columns_to_encode])

print(df.head())

df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df['SuicidalThought'] = df['SuicidalThought'].map({'Yes':1,'No':0})
df['FamilyMental'] = df['FamilyMental'].map({'Yes':1,'No':0})

# Label Encoded
label =LabelEncoder()
df['profession_segment'] = label.fit_transform(df['profession_segment'])
print(df.head())


df['City_segment'].unique()
df['City_segment']=label.fit_transform(df['City_segment'])
df = df.drop(columns=['City','Profession','Degree'])
print(df.head())
df.info()
# Biểu đồ heatmap
X = df.drop(columns=['id']).select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(15,10))
sns.heatmap(X.corr(), cmap='coolwarm', annot=True)
plt.title('Heatmap')
plt.show()
# Vậy tất cả các biến đều ảnh hưởng đến Depression

# boxplot
X = df.drop(columns=['id','Depression'])
Y = df['Depression']
plt.figure(figsize=(12, 6))
sns.boxplot(data=X)  
plt.title('Outlier Dêtct', fontsize=16)
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()  
plt.show()

# Có Age Profession Work Pressure Job Satisfaction và CGPA tồn tại outlier
outlier_columns = ['Age','Work Pressure','Job Satisfaction','profession_segment','CGPA']

def outlierDel(outliers, df):
    for col in outliers:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1 
        lower = Q1 - IQR
        upper = Q3 + IQR 
        df= df[(df[col] >= lower) & (df[col] <= upper)]
    return df
df = outlierDel(outlier_columns,df)
# check
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.drop(columns=['id','Depression']))  
plt.title('Outlier Dêtct', fontsize=16)
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()  
plt.show()

df.reset_index()

# test mô hình

X1 = df.drop(columns=['id','Depression'])
Y1 = df['Depression']

x_train, x_test, y_train, y_test = train_test_split(X1,Y1,test_size= .2, random_state=42)

models = [
    ('LR',LogisticRegression()),
    ('DTC',DecisionTreeClassifier()),
    ('NaiveBayes',GaussianNB()),
    ('KNeighbors',KNeighborsClassifier()),
    ('RFC',RandomForestClassifier()),
    ('SVC',SVC()),
    ('XGBoost',xgb.XGBClassifier()),
]

for i, model in models:
    K = KFold(n_splits=5)
    result = cross_val_score(model, x_train, y_train, cv = K,scoring='accuracy')
    print(i,': ' ,result)

# -> Dung LogisticRegression(chac the)

grid = {
    'penalty':['l1','l2','elasticnet',None],
    'max_iter':[100,150,200],
    'random_state':[5,25]
}
model = LogisticRegression()
grid_search = GridSearchCV(model, grid, scoring='accuracy',cv=5)
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
# l'max_iter': 100, 'penalty': None, 'random_state': 5

model1 = LogisticRegression(penalty= None,max_iter=100,random_state=5)
model1.fit(x_train,y_train)
y_pred_train = model1.predict(x_train)

print("Hệ số (Coefficients):", model1.coef_)

feature_names = X1.columns  
coefficients = model1.coef_[0]  

print("Hệ số tương ứng với từng biến:\n ")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.6f}")

print("Hệ số chặn (Intercept):", model1.intercept_)

# Báo cáo tập train
print('Bao cao ve Logistic Regression \n',classification_report(y_train,y_pred_train))
# tap train chinh xac 0.85
print('Confusion Matrix')
print(ConfusionMatrixDisplay.from_estimator(model1,x_train,y_train))

# tap test
y_pred_test = model1.predict(x_test)
print('Bao cao ve Logistic Regression tap test \n',classification_report(y_test,y_pred_test))

print('Confusion Matrix')
print(ConfusionMatrixDisplay.from_estimator(model1,x_test,y_test))
# tap test chinh xac 0.85 ko overfitting


# tập train
accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)

print(f"train_Accuracy: {accuracy}")
print(f"train_Precision: {precision}")
print(f"train_Recall: {recall}")
print(f"train_F1-Score: {f1}")

# tập test
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f"test_Accuracy: {accuracy}")
print(f"test_Precision: {precision}")
print(f"test_Recall: {recall}")
print(f"test_F1-Score: {f1}")

# powr bi -> highlight
# máy học -> kết luận, dùng được hay ko mức độ hiệu quar.  
# trông trường hợp đã hiệu quả -> có thêm đề xuất cải thiện mô hình
# thêm phần phân chia công việc
















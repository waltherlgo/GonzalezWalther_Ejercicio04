import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
from itertools import combinations

data = pd.read_csv('Cars93.csv')
Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
part=0.5
#Todas las posibles combinaciones
R=[[] for x in range(11)]
maximos=[]
RC=0
for i in range(11):
    com=list(combinations(columns,i+1))
    count=0
    R[i]=np.zeros(len(com))
    for l in com:
        l2=np.array(l)
        X = np.array(data[l2])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=part)
        regresion = sklearn.linear_model.LinearRegression()
        regresion.fit(X_train, Y_train)
        R[i][count]=regresion.score(X_test,Y_test)
        if (R[i][count]>RC):
            RC=R[i][count]
            maximos=l2
        count=count+1
for i in range(11):
    plt.scatter((i+1)*np.ones(R[i].shape[0]),R[i])
plt.title('Máximo con '+str(maximos)+r' $R^2=$ %4.3f'%RC)
plt.xlabel("# de Variables")
plt.ylabel(r'$R^2')
plt.savefig('nparams.png')
plt.show()
print(str(maximos))
X = np.array(data[columns])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=part)
S=np.zeros(20)
count=0
LC=0
betas=[]
for i in np.linspace(-3,1,20):
    alpha=10**i
    lasso = sklearn.linear_model.Lasso(alpha)
    lasso.fit(X_train, Y_train)
    S[count]=lasso.score(X_test, Y_test)
    if S[count]>LC:
        LC=S[count]
        #betas guarda los valores para R^2 máximo
        betas=lasso.coef_
    count=count+1
plt.figure()
plt.plot(np.linspace(-3,1,20),S)
plt.xlabel(r'log $\lambda$')
plt.title('Lasso')
plt.savefig('lasso.png')
plt.show()
print(betas)
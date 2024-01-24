#Jupiter module
#imports
import pandas as pd
import numpy as np


def Kippler(row):
    #applies kippler equation
    jupiter_mass =np.sqrt(((row["distance_km"]**3)*4*(np.pi)**2)/((row["period_days"]**2)*(6.67*10**-11)))
    #returns predicted mass
    return jupiter_mass

#defines Moons class
class Moons:

    def __init__(self):
    
        #defines SQL related variables
        self.connectable = f"sqlite:///jupiter.db"
        self.Query = ("SELECT * FROM moons")

        #reads moons table and stores as data variable
        self.data = pd.read_sql(self.Query, self.connectable, index_col="moon")

        #predicts mass of jupiter using each moon
        self.data["predicted_mass"] = self.data.apply(Kippler, axis=1)

        #removes column with mainly NULL values
        self.data = self.data.drop("mass_kg", axis="columns")

        #creates precalculated columns
        self.data["T^2"] = (self.data["period_days"]*86400)**2
        self.data["a^3"] = (self.data["distance_km"]*1000)**3
    
    def model(self):
        """Creates model"""

        #imports used parts of sklearn
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        #creates linear regression model
        self.model = linear_model.LinearRegression(fit_intercept=True)

        #creates x and y variables
        self.X = self.data[["a^3"]]
        self.Y = self.data["T^2"]

        
        #trains on data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.Y, test_size=0.3, random_state=42)

        #fits model to trained data
        self.model.fit(self.x_train,self.y_train)

        #uses model to predict values
        self.pred_T2 = self.model.predict(self.x_test)

        #outputs R^2 score
        print(f"The R2 score is: {r2_score(self.y_test,self.pred_T2)}")

    def model_plot(self):
        """Plots model data"""

        #imports matplotlib
        import matplotlib.pyplot as plt

        #creates figure
        fig, ax  = plt.subplots()

        #Create a scatter plot of the known a^3-T^2 values
        ax.scatter(self.data["a^3"],self.data["T^2"])

        #Draw line to represent the predicted T^2-values
        ax.plot(self.x_test,self.pred_T2)

        # Axis labels
        ax.set_xlabel("a^3")
        ax.set_ylabel("T^2")

    def summary_stats(self,column, group="group"):
        """Returns summary data of specified group"""

        #groups data and produces summary of specified column
        self.summary = self.data.groupby(group)
        self.summary = self.summary[column].describe()

        return self.summary
    
    def relationship_view(self):
        """outputs plots of each column against each other"""

        from seaborn import pairplot
        
        #produces plot of every column against every column
        pairplot(self.data)
    
    def view(self, head=False):
        """Outputs data or head of data if specified"""

        #returns data or head data
        if head == True:
            return self.data.head()
        else:
            return self.data
        
    def jupiter_mass_predict(self):
        """predicts mass of jupiter"""

        #calculates mass of jupiter using the data from the model
        self.jp_mass = (4* np.pi**2) / (self.model.coef_[0]*6.67e-11)

        #prints f string with predicted mass of jupiter
        print(f"The mass of jupiter is predicted to be {self.jp_mass}Kg")

    def locate(self, items):
        """uses loc to return specified rows"""

        #returns specified rows
        return self.data.loc[items]
    
    def plot(self, x, y, x_label, y_label):
        """Plots specified data"""

        #imports matplotlib
        import matplotlib.pyplot as plt

        #creates figure
        fig, ax  = plt.subplots()

        #Create a scatter plot of the known a^3-T^2 values
        ax.scatter(self.data[x], self.data[y])

        # Axis labels and tile
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    


    
                
    

    



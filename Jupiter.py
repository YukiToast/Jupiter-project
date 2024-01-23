#Jupiter module
import pandas as pd
import numpy as np


def Kippler(row):
    #applies kippler equation
    jupiter_mass =np.sqrt(((row["distance_km"]**3)*4*(np.pi)**2)/((row["period_days"]**2)*(6.67*10**-11)))
    #returns predicted mass
    return jupiter_mass


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

        #MAYBE NOT NECESSARY
        # self.period_days = self.data["period_days"]
        # self.distance_km = self.data["distance_km"]
        # self.radius_km = self.data["radius_km"]
        # self.mag = self.data["mag"]
        # self.mass_kg = self.data["mass_kg"]
        # self.group = self.data["group"]
        # self.ecc = self.data["ecc"]
        # self.inclination_deg = self.data["inclination_deg"]

        #creates precalculated columns
        self.data["T^2"] = (self.data["period_days"]*86400)**2
        self.data["a^3"] = (self.data["distance_km"]*1000)**3

    def list_request(self, request_list):
        """Returns dataframe of items in list"""
        #allows list of moon names to be taken from data and returned
        temp_df = self.data["moon"]
        for i in range(len(request_list)):
            temp_df.join(self.data[request_list[i]])
        return temp_df
    
    
    def model(self):
        """Creates model of data"""
        #imports used parts of sklearn
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

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

        #outputs R^2 score and RMSE score
        print(f"The R2 score is: {r2_score(self.y_test,self.pred_T2)}")
        print(f"The RMSE score is: {mean_squared_error(self.y_test,self.pred_T2, squared=True)}")

    def plot(self):
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

    def summary_stats(self, group):
        """Returns summary data of specified group"""

        self.summary = self.data.groupby(group).describe()

        return self.summary
    
    def relationship_view(self):

        from seaborn import pairplot
        
        pairplot(self.data)
    
    def review_data(self):
        return self.data

    
                
    

    



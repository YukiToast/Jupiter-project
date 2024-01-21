#Jupiter module
import pandas as pd
import numpy as np

def Kippler(row):
    jupiter_mass =np.sqrt(((row["distance_km"]**3)*4*(np.pi)**2)/((row["period_days"]**2)*(6.67*10**-11)))

    return jupiter_mass


class Moons():

    def __init__(self):
        connectable = f"sqlite:///jupiter.db"
        Query = ("SELECT *")

        self.data = pd.read_sql(Query, connectable, index_col="moon")
        self.data["predicted_mass"] = self.data.apply(Kippler, axis=1)

        self.name = data["moon"]
        self.period_days = data["period_days"]
        self.distance_km = data["distance_km"]
        self.radius_km = data["radius_km"]
        self.mag = data["mag"]
        self.mass_kg = data["mass_kg"]
        self.group = data["group"]
        self.ecc = data["ecc"]
        self.inclination_deg = data["inclination_deg"]

        self.data["T^2"] = (self.data["period_days"]*86400)**2
        self.data["a^3"] = (self.data["distance_km"]*1000)**3

    def output(self, request_list):
        temp_df = self.data["moon"]
        for i in range(len(request_list)):
            temp_df.join(self.data[request_list[i]])
        return temp_df
    
    def model(self):
    
        from sklearn import linear_model

        self.model = linear_model.LinearRegression(fit_intercept=True)

        self.X = self.moons_df[["a^3"]]
        self.Y = self.moons_df["T^2"]

        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.Y, test_size=0.3, random_state=42)


        self.model.fit(x_train,y_train)

        self.pred_T2 = self.model.predict(x_test)

        from sklearn.metrics import mean_squared_error, r2_score

        print(f"The R2 score is: {r2_score(self.y_test,self.pred_T2)}")
        print(f"The RMSE score is: {mean_squared_error(self.y_test,self.pred_T2, squared=True)}")

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax  = plt.subplots()

        #Create a scatter plot of the known a^3-T^2 values
        ax.scatter(moons_df["a^3"],moons_df["T^2"])

        #Draw line to represent the predicted T^2-values
        ax.plot(x_test,pred_T2)

        # Axis labels
        ax.set_xlabel("a^3")
        ax.set_ylabel("T^2")
                
    

    



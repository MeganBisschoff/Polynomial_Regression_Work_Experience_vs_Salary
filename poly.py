# ----- Polynomial Regression ----- #
'''
This is a polynomial regression on Work Experience Years vs Salary. The relationship between the independent variable x (Work Experience Years) and the dependent variable y (Monthly Salary) is modelled as an nth degree polynomial to accurately predict my salary as a Software Engineer after 5 years of work experience.

* No testing data as the set is too small
'''

# ---------- Import Libraries ---------- #
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------- Initialise Data ---------- # 

# Initialise the 'years' of experience array as x data and reshape to 2D array.
years = np.array([2, 4, 6, 8, 10]).reshape((-1, 1)) 
# Initialise the 'salary' array as y data   
salaries = np.array([26867, 38360, 49049, 61178, 81279]) 

# ---------- Linear Regression model ---------- #

# Initialise LR object and fit the training data to it.
lin_model = LinearRegression()
lin_model.fit(years, salaries)

# Plot the linear regression results, in dotted blue line, of the years to the predicted salaries.
plt.plot(years, lin_model.predict(years), 'b:') 

# ---------- Polynomial Regression model ---------- #

# Initialise PR model and transform the training data with formula y = ß0 + ß1x + ß2x2 to the second degree to solve for the coefficient.
# Save instance of PR features for further fitting and predicting.
poly_model = PolynomialFeatures(degree = 2)
trans_years = poly_model.fit_transform(years)

# Initialise LR model and fit to the transformed polynomial terms.
poly_lin_model = LinearRegression().fit(trans_years, salaries)

# Plot the polynomial regression results, in dashed red line, of the years to the predicted salaries.
plt.plot(years, poly_lin_model.predict(poly_model.fit_transform(years)), 'r--') 

# ---------- Salary Prediction ---------- #

# Predict and plot the 5 year salary, with a blue star, using the linear model.
pred_lin_salary = lin_model.predict([[5]])
plt.plot(5, pred_lin_salary, "b*", markersize=10)

# Predict and plot the 5 year salary, with a red star, using the polynomal model.
pred_poly_salary = poly_lin_model.predict(poly_model.fit_transform([[5]]))
plt.plot(5, pred_poly_salary, "r*", markersize=10)

# ---------- Graph Annotations ---------- #

# Scatter the years and salries data points.
plt.scatter(years, salaries, c='g')

# Annotate the title, labels and legend.
plt.title('-- My 5 Year Software Engineering Salary Forecast --')
plt.xlabel('Work Experience (years)')
plt.ylabel('Salary (ZAR)')
plt.legend(["Linear model", "Polynomial model", "Linear predicted salary", "Polynomial predicted salary", "Training Data"])
plt.grid(True, linewidth=0.1)

# Annotate each x,y coordinate positions, offset by 0.5 for legibility.
for index in range(len(salaries)):
    plt.text(years[index] +0.5, salaries[index], 'R ' +str(salaries[index]), color='g')

# Annotate the linear and poly predicted x,y coordinate positions, offset by 1.5
plt.text(5 -1.5, pred_lin_salary, 'R ' +str(*pred_lin_salary), color='b')
plt.text(5 -1.5, pred_poly_salary, 'R ' +str(*pred_poly_salary), color='r')

# Save and show graph.
plt.savefig("Years_Salaries.png")
plt.show()

# ---------- Prediction Results ---------- #
print(f"\nMy monthly salary forecast after 5 years of experience is R", (np.round(*pred_poly_salary,2)))


# ---------- Resources ---------- #
'''
# For salary data:
    - Nel, J. (2021) Developer Salaries 2021: Cape Town, Johannesburg and Pretoria. 
    https://www.offerzen.com/blog/developer-salaries-in-cape-town-vs-johannesburg
    - Software Engineer average salary in South Africa, 2022. https://za.talent.com/salary?job=software+engineer#:~:text=The%20average%20software%20engineer%20salary%20in%20South%20Africa%20is%20R,7%20800%20000%20per%20year.
# For understanding steps required:
    - Agrawal, R. (2022) All you need to know about Polynomial Regression. 
    https://www.analyticsvidhya.com/blog/2021/07/all-you-need-to-know-about-polynomial-regression/
    - Bhardwaj, A. (2021)  Position salary — polynomial regression. 
    https://becominghuman.ai/position-salary-polynomial-regression-dd6af8028a95
    - Ujhelyi, T. Polynomial Regression in Python using scikit-learn. 
    https://data36.com/polynomial-regression-python-scikit-learn/
    - Stojiljković, M. (2021) Linear Regression in Python. 
    https://realpython.com/linear-regression-in-python/#simple-linear-regression-with-scikit-learn
# For printing raw data from an array
    - https://www.geeksforgeeks.org/python-star-or-asterisk-operator/
# For plotting text:
    - https://discord.com/channels/577460436316717056/1042035662523740160/1042818383789436958
# For matplotlib shorthand:
    - https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html

'''
import matplotlib.pyplot as plt

##"""
##-- Line Graph --##

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
temps = [30, 32, 31, 29, 28, 27, 26]
plt.plot(days, temps, marker="o")
plt.title("Weekly Temperatures")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()

#-- Bar Chart --#

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
productA = [200, 240, 210, 300, 280]
plt.bar(months, productA, color='blue')
plt.title("Product A Sales Over 5 Months")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

#-- Multi Line Graph --#

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
apple = [120, 125, 123, 130, 128, 132, 129]
google = [1000, 1005, 1010, 1020, 1015, 1030, 1025]
amazon = [1800, 1810, 1790, 1820, 1815, 1830, 1825]
plt.plot(days, apple, label='Apple')
plt.plot(days, google, label='Google')
plt.plot(days, amazon, label='Amazon')
plt.legend()
plt.title("Stock Prices Over a Week")
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()

#-- Bar Chart Horizontal --#

cities = ['City A', 'City B', 'City C', 'City D']
population = [150000, 120000, 180000, 100000]
plt.barh(cities, population, color='orange')
plt.title("Population by City")
plt.xlabel("Population")
plt.ylabel("City")
plt.show()

#-- Pie Chart --#

brands = ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Others']
share = [30, 25, 20, 10, 15]
plt.pie(share, labels=brands, autopct="%1.1f%%", startangle=90)
plt.title("Smartphone Market Share")
plt.axis('equal')
plt.show()

#-- Histogram --#

scores = [56, 67, 45, 88, 72, 90, 61, 76, 84, 69]
plt.hist(scores, bins=5, color='green', edgecolor='black')
plt.title("Score Distribution")
plt.xlabel("Scores")
plt.ylabel("Number of Students")
plt.show()

#-- Scatter Plot --#

height = [150, 155, 160, 165, 170, 175]
weight = [45, 50, 55, 60, 65, 70]
plt.scatter(height, weight, color='red')
plt.title("Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.grid()
plt.show()

#-- Area Plot --#

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
expenses = [2000, 2200, 2100, 2500, 2400]
plt.fill_between(months, expenses, color='skyblue', alpha=0.5)
plt.plot(months, expenses, color='blue')
plt.title("Monthly Expenses")
plt.xlabel("Month")
plt.ylabel("Expense (INR)")
plt.show()

#-- Bubble Chart --#
x = [10, 20, 30, 40, 50]
y = [15, 25, 35, 45, 55]
sizes = [100, 300, 500, 700, 900]
plt.scatter(x, y, s=sizes, alpha=0.5, c='purple')
plt.title("Bubble Chart Example")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

#"""

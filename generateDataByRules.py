# Data description:
# 4 attributes (k = 4), number of data = 1000 (M = 1200)
#
# Absolutely 'right' Rules: 
# 1. If 0 <= time <= 5, then class = No
# 2. If outlook == Rainy, then class = No
# 3. If temperature < 9 or temperature > 32, then class = No
# 4. If temperature > 30 and humidity > 0.85, then class = No
# 5. If temperature < 12 and outlook == Windy, then class = No
# 6. Otherwise class = Yes


import numpy as np

if __name__=="__main__":
    with open("./Data/data.csv", 'w', encoding = 'UTF-8') as f:
        M = 1200
        np.random.seed(0)
        demo_list = ['Rainy','Sunny','Cloudy','Windy','Overcast']
        outlook = np.random.choice(demo_list, size=M, p=[0.15,0.35,0.3,0.15,0.05])
        humidity = np.random.rand(M)
        temperature = np.random.randint(5, 36, size=M)
        time = np.random.randint(0, 24, size=M)
        f.write("outlook,humidity,temperature,time,hangingOut\n")
        for i in range(0, M):
            f.write("%s," % outlook[i])
            if outlook[i] == 'Rainy':
                humidity[i] = 1
            f.write("%.4f," % humidity[i])
            f.write("%d," % temperature[i])
            f.write("%d," % time[i])
            # Rules:
            if 0 <= time[i] <= 5:
                f.write("No\n")
            elif outlook[i] == 'Rainy':
                f.write("No\n")
            elif temperature[i] < 9 or temperature[i] > 32:
                f.write("No\n")
            elif temperature[i] > 30 and humidity[i] > 0.85:
                f.write("No\n")
            elif temperature[i] < 12 and outlook[i] == 'Windy':
                f.write("No\n")
            else:
                f.write("Yes\n")
    f.close()
f = open("out.txt", "r")
no_of_vehicles=[]
no_of_vehicles.append(int(f.readline()))
no_of_vehicles.append(int(f.readline()))
no_of_vehicles.append(int(f.readline()))
no_of_vehicles.append(int(f.readline()))

baseTimer = 120  # baseTimer = int(input("Enter the base timer value"))
timeLimits = [5, 30]  # timeLimits = list(map(int,input("Enter the time limits ").split()))

print("Input no of vehicles : ", *no_of_vehicles)
t = [(i / sum(no_of_vehicles)) * baseTimer if timeLimits[0] < (i / sum(no_of_vehicles)) * baseTimer < timeLimits[1] 
else min(timeLimits, key=lambda x: abs(x - (i / sum(no_of_vehicles)) * baseTimer)) for i in no_of_vehicles]

print(t, sum(t))
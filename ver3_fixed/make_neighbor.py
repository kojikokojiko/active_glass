
import numpy as np
def makeneighborList_dynamic(aroundGridNum):
    neighbor_row=[]
    neighbor_col=[]
#xの近接リスト
    neighborRowValue = -aroundGridNum
    zero_count=0
  # countIndexRow = 0
    for i in range(aroundGridNum*2+1):
        for j in range(aroundGridNum*2+1):
            if neighborRowValue==0:
                zero_count+=1
                if zero_count==aroundGridNum+1:
                    continue
      # neighbor_row[countIndexRow]=neighborRowValue
            neighbor_row.append(neighborRowValue)
      # countIndexRow+=1
        neighborRowValue+=1

# yの近接リスト
    zero_count=0
    neighborColValue=-aroundGridNum
    for i in range(aroundGridNum*2+1):
        for j in range(aroundGridNum*2+1):
            if neighborColValue==0:
                zero_count+=1
                if zero_count==aroundGridNum+1:
                    print(neighborColValue)
                    print("jjjjj")
                    neighborColValue+=1

                    continue
            neighbor_col.append(neighborColValue)
            neighborColValue+=1
        neighborColValue=-aroundGridNum


    return neighbor_row,neighbor_col


row,col=makeneighborList_dynamic(4)
print(row)
print(col)


# [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
#  -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
#  -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
#  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
#   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
#    0,  0,  0,  0,  0,      0,  0,  0,  0,  0, 
#    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
#     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]



# [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 
#  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
#   -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
   -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 
   -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 
   -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1,    1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
 -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


 [-4, -4, -4, -4, -4, -4, -4, -4, -4,
  -3, -3, -3, -3, -3, -3, -3, -3, -3,
  -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  0,  0,  0,      0,  0,  0,  0, 
   1,  1,  1,  1,  1,  1,  1,  1,  1,
      2, 2, 2, 2, 2, 2, 2, 2, 2,
       3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4]

[-4, -3, -2, -1, 0, 1, 2, 3, 4,
 -4, -3, -2, -1, 0, 1, 2, 3, 4,
 -4, -3, -2, -1, 0, 1, 2, 3, 4,
 -4, -3, -2, -1, 0, 1, 2, 3, 4,
 -4, -3, -2, -1,    1, 2, 3, 4,
 -4, -3, -2, -1, 0, 1, 2, 3, 4,
 -4, -3, -2, -1, 0, 1, 2, 3, 4,
 -4, -3, -2, -1, 0, 1, 2, 3, 4,
 -4, -3, -2, -1, 0, 1, 2, 3, 4]
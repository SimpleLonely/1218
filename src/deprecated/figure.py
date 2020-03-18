from functools import reduce

init = '''[ 14.62165]
 [ 29.74631]
 [  7.01544]
 [ 29.37105]
 [  7.72766]
 [  8.17111]
 [-10.32994]
 [ -8.83611]
 [-17.08779]
 [-21.8537 ]
 [ -5.22213]
 [-60.83104]
 [-18.68213]
 [  8.6662 ]
 [ -4.0162 ]
 [ 58.27276]
 [ -47.86519]
 [ 17.34739]
 [ -9.48414]
 [ -53.82614]'''
 exacX = 6.15
result_list = [float(i[2:-1].strip())+exacX for i in init.splitlines()]
result_list = [x*x for x in result_list]
product = reduce((lambda x, y: x + y), result_list)/len(result_list)
print (product)
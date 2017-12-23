# Imports:
import numpy as np

# Definicion de los kernels:

#---------------------------------------------------------------------------
# linear_kernel(x, y, b=1)
#   Calcula el kernel lineal de x con y.
# Argumentos:
#   x: array de numpy, dimensiones n x d
#   y: array de numpy, dimensiones m x d
#   b: bias, por defecto es 1
# Devuelve:
#   Array k de dimensiones n x m, con kij = k(x[i], y[j])
#---------------------------------------------------------------------------
def linear_kernel(x, y, b=1):
    
    K=np.dot(x, y.T) + b
    
    return K

#---------------------------------------------------------------------------
# poly_kernel(x, y, deg=1, b=1)
#   Calcula kernels polinomicos de x con y, k(x, y) = (xy + b)^deg
# Argumentos:
#   x: array de numpy, dimensiones n x d
#   y: array de numpy, dimensiones m x d
#   deg: grado, por defecto es 1
#   b: bias, por defecto es 1
# Devuelve:
#   Array K de dimensiones n x m, con Kij = k(x[i], y[j])
#---------------------------------------------------------------------------
def poly_kernel(x, y, deg=1, b=1):
    
    K=np.power(np.dot(x, y.T)+b, deg)
    
    return K


#---------------------------------------------------------------------------
# rbf_kernel(x, y, sigma=1)
#   Calcula kernels gausianos de x con y, k(x, y) = exp(-||x-y||^2 / 2*sigma^2)
# Argumentos:
#   x: array de numpy, dimensiones n x d
#   y: array de numpy, dimensiones m x d
#   deg: anchura del kernel, por defecto es 1
# Devuelve:
#   Array K de dimensiones n x m, con Kij = k(x[i], y[j])
#---------------------------------------------------------------------------
def rbf_kernel(x, y, sigma=1):
    
    numerador=np.power(np.linalg.norm(x, axis=x.ndim-1, keepdims=True),2) + np.power((np.linalg.norm(y,axis=y.ndim-1, keepdims=True).T),2) - 2 * np.dot(x, y.T)
    denominador=2*np.power(sigma,2)
    K=np.exp((-numerador)/denominador)
    
    return K


#---------------------------------------------------------------------------
# Clase SVM:
#   C: parametro de complejidad (regularizacion)
#   kernel_params: diccionario con los parametros del kernel
#     -- "kernel": tipo de kernel, puede tomar los valores "linear", "poly"
#                  y "rbf" 
#     -- "sigma": (solo para kernel gausiano) anchura del kernel 
#     -- "deg": (solo para kernel polinomico) grado del kernel
#     -- "b": (solo para kernel lineal y polinomico) bias
#   alpha: multiplicadores de Lagrange
#   b: bias
#   X: array de atributos, dimensiones (n, d)
#   y: array de clases, dimensiones (n,)
#   is_sv: array de booleanos que indica cuales de los vectores son de
#          soporte
#   num_sv: numero de vectores de soporte
#---------------------------------------------------------------------------
class SVM:
    def __init__(self, C=1, kernel="rbf", sigma=1, deg=1, b=1):
        self.C = C
        self.kernel_params = {"kernel": kernel, "sigma": sigma, "deg": deg, "b": b}
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.is_sv = None
        self.num_sv = 0

    #-----------------------------------------------------------------------
    # evaluate_kernel(self, x, y)
    #   Evalua el kernel sobre los arrays x e y
    # Argumentos:
    #   x: array de numpy, dimensiones n x d
    #   y: array de numpy, dimensiones m x d
    # Devuelve:
    #   Array de dimensiones n x m
    #-----------------------------------------------------------------------
    def evaluate_kernel(self, x, y):
        k = self.kernel_params["kernel"]
        sigma = self.kernel_params["sigma"]
        deg = self.kernel_params["deg"]
        b = self.kernel_params["b"]
        
        if k == "linear":
            return linear_kernel(x, y, b)
        if k == "poly":
            return poly_kernel(x, y, deg, b)
        if k == "rbf":
            return rbf_kernel(x, y, sigma)

    #-----------------------------------------------------------------------
    # init_model(self, a, b, X, y)
    #   Inicializa las alphas y el b del modelo a los valores pasados como
    #   argumentos. Inicializa la X y la y del modelo a los valores pasados
    #   como argumentos.
    # Argumentos:
    #   a: array de alphas
    #   b: bias
    #   X: array de atributos
    #   y: array de clases
    #-----------------------------------------------------------------------
    def init_model(self, a, b, X, y):
        self.alpha = a
        self.is_sv = self.alpha > 0
        self.num_sv = np.sum(self.is_sv)
        self.b = b
        self.X = X
        self.y = y
        
    #-----------------------------------------------------------------------
    # evaluate_model(self, z)
    #   Evalua el modelo sobre un conjunto de datos z, devuelve f(z).
    # Argumentos:
    #   z: array de numpy, dimensiones (n, d)
    # Devuelve:
    #   f, array de numpy, dimensiones (n, 1)
    #-----------------------------------------------------------------------
    def evaluate_model(self, z):
        n, d = z.shape
        f = np.zeros(n)

        # f(z) = Sum1->n (Alphai * Yi * K(Zi,z) + b)
        f = np.sum(self.alpha * self.y * self.evaluate_kernel(self.X, z), axis=1) + self.b
        
        return f

    #-----------------------------------------------------------------------
    # select_alphas(self, e, tol)
    #   Selecciona las dos alphas a optimizar de manera heuristica. Primero
    #   evalua las restricciones sobre todas las alphas, y busca la primera
    #   que no las satisface. Luego elige una segunda alpha al azar
    #   distinta de la primera.
    #   Devuelve los indices de las dos alphas elegidas. Si todas las alphas
    #   satisfacen las restricciones devuelve -1 como indices.
    # Argumentos:
    #   e: error para cada x, ei = f(xi) - yi
    #   tol: tolerancia maxima permitida para la satisfaccion de las
    #        restricciones
    # Devuelve:
    #   i: indice de la primera alpha seleccionada
    #   j: indice de la segunda alpha seleccionada
    #   (NOTA: si todas las alphas satisfacen las restricciones no hay que
    #   devolver ningun indice, en este caso la funcion devuelve i = j = -1)
    #-----------------------------------------------------------------------
    def select_alphas(self, e, tol):
        n = e.shape[0]
        ye = self.y*e
        a = self.alpha
        C = self.C
        ix = np.ones(n, dtype=bool)
        
        ix = np.asarray([(ye[i] < - tol and a[i] < C) or (ye[i] > tol and a[i] > 0) for i in range(n)])
        
        # Si todas las alphas satisfacen las restricciones, devuelvo i = j = -1:
        if np.sum(ix) == 0:
            return -1, -1

        # Cojo como i el indice de la primera que no satisface las restricciones:
        i = (ix*range(n))[ix][0]
        
        # Cojo como j otro al azar distinto de i:
        p = np.random.permutation(n)[:2]        
        j = p[0] if p[0] != i else p[1]
        
        return i, j
        
    #-----------------------------------------------------------------------
    # calculate_eta(self, z)
    #   Calcula el numero eta (denominador) que aparece en el algoritmo SMO.
    # Argumentos:
    #   z: array de dimension (2, d) que contiene las x asociadas a las dos
    #      alphas seleccionadas en el paso anterior del algoritmo
    # Devuelve:
    #   eta: valor que aparece en el denominador de la eq. 16 en el articulo
    #        de Platt, 1998.
    #-----------------------------------------------------------------------
    def calculate_eta(self, z):
        eta = 0
        
        x1 = z[0]
        x2 = z[1]
                
        eta = self.evaluate_kernel(x1,x1) + self.evaluate_kernel(x2,x2) - (2 * self.evaluate_kernel(x1,x2))

        return eta

    #-----------------------------------------------------------------------
    # update_alphas(self, i, j, eta, e)
    #   Actualiza los valores de las dos alphas seleccionadas, devuelve los
    #   valores antiguos.
    # Argumentos:
    #   i: indice de la primera alpha
    #   j: indice de la segunda alpha
    #   eta: valor del denominador de la eq. 16 del articulo de Platt, 1998
    #   e: error para cada x, ei = f(xi) - yi
    # Devuelve:
    #   ai_old: valor antiguo de alpha_i
    #   aj_old: valor antiguo de alpha_j
    #-----------------------------------------------------------------------
    def update_alphas(self, i, j, eta, e):
        ai_old = self.alpha[i]
        aj_old = self.alpha[j]
        E1 = e[i]
        E2 = e[j]
        y1 = self.y[i]
        y2 = self.y[j]
        
        #-------------------------------------------------------------------
        # 1. Calcula los valores minimo y maximo (L y H) que puede tomar
        #    alpha_j
        L = H = 0
        if y1 == y2:
            L = max(0, aj_old + ai_old - self.C)
            H = min(self.C, aj_old + ai_old)
        else:
            L = max(0, aj_old - ai_old)
            H = min(self.C, self.C + aj_old - ai_old)
        
        # 2. Calcula el nuevo valor de alpha_j segun la ecuacion 16 del
        #    articulo de Platt, 1998
        aj = aj_old + np.divide(y2 * (E1-E2), eta)
        
        # 3. Haz el clip de alpha_j para que este en el rango [L, H]
        aj = np.clip(aj,L,H)
        
        # 4. Calcula el nuevo valor de alpha_i con la ecuacion 18
        ai = ai_old + y1 * y2 * (aj_old - aj)
        #-------------------------------------------------------------------    
        
        self.alpha[i] = ai
        self.alpha[j] = aj

        self.is_sv[i] = self.alpha[i] > 0
        self.is_sv[j] = self.alpha[j] > 0
        self.num_sv = np.sum(self.is_sv)
        
        return ai_old, aj_old

    #-----------------------------------------------------------------------
    # update_b(self, i, j, ai_old, aj_old, e)
    #   Actualiza el valor del bias.
    # Argumentos:
    #   i: indice de la primera alpha
    #   j: indice de la segunda alpha
    #   ai_old: valor antiguo de alpha_i
    #   aj_old: valor antiguo de alpha_j    
    #   e: error para cada x, ei = f(xi) - yi
    # Devuelve:
    #   Nada.
    #-----------------------------------------------------------------------
    def update_b(self, i, j, ai_old, aj_old, e):
        E1 = e[i]
        E2 = e[j]
        y1 = self.y[i]
        y2 = self.y[j]
        ai = self.alpha[i]
        aj = self.alpha[j]
        X1 = self.X[i]
        X2 = self.X[j]
        
        p1 = y1 * (ai - ai_old) 
        p2 = y2 * (aj - aj_old)
        
        b1 = self.b - E1 - p1 * self.evaluate_kernel(X1,X1) - p2 * self.evaluate_kernel(X1,X2)
        b2 = self.b - E2 - p1 * self.evaluate_kernel(X1,X2) + p2 * self.evaluate_kernel(X2,X2)
        
        if 0 < ai < self.C:
            self.b = b1 
        elif 0 < aj < self.C:
            self.b = b1
        else:
            self.b = (b1 + b2)/2
        
        
    #-----------------------------------------------------------------------
    # simple_smo(self, X, y, tol=0.00001, maxiter=10, verb=False)
    #   Ejecuta el algoritmo SMO (version simplificada) sobre los datos
    #   (X, y).
    # Argumentos:
    #   X: array de atributos
    #   y: array de clases
    #   tol: tolerancia para el grado de satisfaccion de las restricciones
    #        de las alphas, por defecto es 0.00001
    #   maxiter: maximo numero de iteraciones, por defecto es 10
    #   verb: flag booleano para mostrar info (True) o no (False), por
    #         defecto es False
    #   print_every: entero que indica cada cuantas iteraciones se muestra
    #                informacion
    #-----------------------------------------------------------------------
    def simple_smo(self, X, y, tol=0.00001, maxiter=10, verb=False, print_every=1):
        n, d = X.shape
        num_iters = 0
        
        # Inicializamos el modelo con las alphas y el bias a 0:
        self.init_model(np.zeros(n), 0, X, y)
        
        # Iteramos hasta maxiter:
        while num_iters < maxiter:
            num_iters += 1
            # Calculamos los errores:
            f = self.evaluate_model(X)
            e = f - y
                        
            # Seleccionamos pareja de alphas para optimizar:
            i, j = self.select_alphas(e, tol)
            
            # Si todas las alphas satisfacen las restricciones, acabamos:
            if i == -1:
                break

            # Calculamos eta:
            eta = self.calculate_eta(X[[i,j],:])

            # Si eta es negativa o cero ignoramos esta pareja de alphas:
            if eta <= 0:
                continue
           
            # Actualizamos las alphas:
            ai_old, aj_old = self.update_alphas(i, j, eta, e)

            # Si no ha habido cambio importante en las alphas, continuamos 
            # sin actualizar el bias:
            if abs(self.alpha[j] - aj_old) < tol:
                continue
                            
            # Actualizamos el bias:
            self.update_b(i, j, ai_old, aj_old, e)

            # Incrementamos el contador de iteraciones e imprimimos:
            if verb and num_iters%print_every == 0:
                print ("Iteration (%d / %d), num. sv: %d" % (num_iters, maxiter, self.num_sv))
            
            


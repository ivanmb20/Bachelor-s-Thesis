compute_net_forces_parallel toma un array de posiciones (N,2) y devuelve las fuerzas netas en cada partícula (N,2) acumulando las fuerzas pairwise calculadas por pairwise_lennarJonnes_gu.
# Explicación paso a paso

* Entrada esperada: bodies — numpy array shape (N,2), dtype float64 (posiciones x,y).
* forces = np.zeros((num_bodies, 2), dtype=np.float64): crea el arreglo de salida donde se acumularán fuerzas por partícula.
* for i in range(num_bodies): itera sobre cada partícula i.
    other_bodies = np.delete(bodies, i, axis=0): crea un array con todas las partículas excepto i (alocación costosa).
    assert ...: comprobación de seguridad sobre la forma de other_bodies.
    current_body = np.tile(bodies[i], (num_bodies-1, 1)): replica la posición de i para emparejarla con other_bodies (otra alocación).
    forces_on_i = pairwise_lennarJonnes_gu(current_body, other_bodies, np.zeros_like(other_bodies)):
    llama a la función guvectorize que calcula la fuerza entre cada par (i, j) y devuelve un array (num_bodies-1, 2) con fuerzas ejercidas sobre i por cada otro j. pairwise_lennarJonnes_gu implementa corte por distancia, epsilon, rMin/rInt y la expresión del potencial.
    forces[i, :] = np.sum(forces_on_i, axis=0): suma todas las contribuciones y guarda la fuerza neta sobre i.
    return forces: devuelve el array (N,2).

## **Observaciones importantes**

* Complejidad: O(N^2) en tiempo. Además, dentro del bucle hay muchas asignaciones temporales (np.delete, np.tile, np.zeros_like) que generan overhead de memoria y ralentizan la ejecución.
* pairwise_lennarJonnes_gu ya aplica cortes (distancias > rInt → fuerza 0) y evita divisiones por cero; eso reduce cómputo efectivo pero no las alocaciones.
* Salida y unidades: fuerzas en las unidades definidas en pairwise_lennarJonnes_gu (mismas unidades que posiciones y constantes G,H). Asegúrate de que bodies sea float64 para mantener compatibilidad con la función guvectorize.


## **Sugerencias de mejora (rendimiento / memoria)**

* Evitar np.delete y np.tile: calcular pares (i,j) con doble bucle y acumular simétricamente (force_ij sobre i y -force_ij sobre j) para ahorrar la mitad del trabajo.
* Implementar con numba.njit para loops en Python y eliminar alocaciones intermedias.
* Si N grande, usar una grilla espacial / cell list para aplicar cutoff rInt y reducir pares evaluados.
Código sugerido (reemplazo más eficiente usando symmetría + numba): pégalo en tu notebook en lugar de la función actual.

### Código sugerido

```

from numba import njit

@njit
def pairwise_force_scalar(x1, y1, x2, y2, G, H, p, q, rInt, rMin):
    dx = x2 - x1
    dy = y2 - y1
    distance_sq = dx*dx + dy*dy
    if distance_sq < 1e-12 or distance_sq > rInt*rInt:
        return 0.0, 0.0
    dist = m.sqrt(distance_sq)
    if dist < rMin:
        dist = rMin
    fx_dir = dx/dist
    fy_dir = dy/dist
    # reproduce the potential logic from pairwise_lennarJonnes_gu
    upperPot = -G
    tmp = 1.0
    i = 1
    tmp *= dist
    while i < p - q:
        i += 1
        tmp *= dist
    upperPot += H*tmp
    while i < p:
        i += 1
        tmp *= dist
    fx = fx_dir * upperPot / tmp
    fy = fy_dir * upperPot / tmp
    return fx, fy

@njit
def compute_net_forces(bodies, G, H, p, q, rInt, rMin):
    N = bodies.shape[0]
    forces = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        xi = bodies[i,0]; yi = bodies[i,1]
        for j in range(i+1, N):
            xj = bodies[j,0]; yj = bodies[j,1]
            fx, fy = pairwise_force_scalar(xi, yi, xj, yj, G, H, p, q, rInt, rMin)
            # accumulate: force on i plus, on j minus (action = -reaction)
            forces[i,0] += fx; forces[i,1] += fy
            forces[j,0] -= fx; forces[j,1] -= fy
    return forces
```

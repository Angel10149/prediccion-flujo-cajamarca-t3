# Clasificación cuántica mediante Q-SVM

## Fundamentos cuánticos
Un qubit tiene un estado normalizado α|0⟩ + β|1⟩. La medición produce 0 o 1 con probabilidades determinadas por los cuadrados de las amplitudes. La superposición permite transformar amplitudes mediante operaciones unitarias. El entrelazamiento aparece cuando el estado de varios qubits no puede expresarse como producto de estados individuales. Estas propiedades permiten construir representaciones de datos con correlaciones complejas.

## SVM clásico
Una máquina de vectores de soporte busca una frontera de clasificación con margen amplio. El parámetro C regula el compromiso entre margen y errores de entrenamiento. Un kernel calcula similitudes entre observaciones y permite representar fronteras no lineales. En un SVM con kernel RBF se ajustan C y gamma usando exclusivamente el entrenamiento y su validación.

## Versión cuántica
En un clasificador con kernel cuántico se codifica cada vector x en el estado |φ(x)⟩ mediante un circuito U(x). La similitud se calcula como K(x,z)=|⟨φ(x)|φ(z)⟩|². La matriz de similitudes alimenta un SVM clásico. La estimación del kernel es la parte que puede ejecutarse en un procesador cuántico. Havlíček y colaboradores presentan este enfoque y un clasificador variacional como vías para explorar espacios de características cuánticos.

## Ventaja y desafíos
Una representación cuántica difícil de simular clásicamente podría resultar útil si su similitud coincide con la estructura de clasificación del problema. La dimensión del espacio de estados por sí sola no demuestra mejora predictiva ni aceleración. Deben medirse carga de datos, preparación de circuitos, número de ejecuciones y coste de construir el kernel. Para N ejemplos se requieren del orden de N² comparaciones. El ruido, la decoherencia y la conectividad restringen la profundidad útil de los circuitos en dispositivos NISQ. El tamaño limitado de los datos, la selección del mapa de características y la comparación con kernels clásicos fuertes condicionan la utilidad práctica.

## Herramientas y simulación propuesta
Se propone Qiskit con Statevector como simulador ideal local. Se usarían dos características estandarizadas a partir de entrenamiento, dos qubits y un mapa con rotaciones dependientes de los datos y una puerta entrelazante. Las fidelidades entre estados formarían las matrices de entrenamiento y prueba para SVC con kernel precomputado. Una comparación usaría SVM lineal, RBF y cuántico, la misma partición estratificada y búsqueda de hiperparámetros con validación interna. Se reportarían F1, accuracy, matriz de confusión, tiempo total y variación entre particiones. Esta es una propuesta experimental y las métricas quedan pendientes de ejecución.

## Conclusión
Q-SVM permite investigar nuevas funciones de similitud. Su valor debe determinarse mediante experimentos comparables y costes completos. Un simulador ideal facilita verificar el circuito, aunque una ventaja práctica exige resultados en condiciones realistas.

## Referencias
Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature, 567, 209–212. https://doi.org/10.1038/s41586-019-0980-2

IBM Quantum. (s. f.). Quantum kernel training. https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-kernel-training

IBM Quantum. (s. f.). quantum_info. https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.5/quantum_info

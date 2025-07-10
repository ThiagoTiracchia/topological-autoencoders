import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def obtener_metricas(carpeta_principal):
    """
    Recorre todas las subcarpetas de la carpeta principal y recopila los datos de los archivos metrics.json
    
    Args:
        carpeta_principal (str): Ruta de la carpeta principal que contiene las subcarpetas con metrics.json
    
    Returns:
        list: Lista de diccionarios con los datos de cada metrics.json
    """
    metricas = []
    
    # Convertir a path absoluto (resuelve problemas con Docker)
    carpeta_absoluta = os.path.abspath(carpeta_principal)
    
    if not os.path.exists(carpeta_absoluta):
        print(f"Error: La carpeta no existe: {carpeta_absoluta}")
        return metricas
    
    if not os.path.isdir(carpeta_absoluta):
        print(f"Error: La ruta no es una carpeta: {carpeta_absoluta}")
        return metricas
    
    for root, dirs, files in os.walk(carpeta_absoluta):
        if 'metrics.json' in files:
            ruta_metrics = os.path.join(root, 'metrics.json')
            nombre_subcarpeta = os.path.basename(root)
            
            try:
                with open(ruta_metrics, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                    metricas.append((nombre_subcarpeta, ruta_metrics, datos))
            except Exception as e:
                print(f"Error al leer {ruta_metrics}: {str(e)}")
    
    return metricas

def extraer_datos(metricas,categoria='training',subcategoria='loss'):
    """
    Extrae los datos de training.loss.mean de la lista de métricas
    
    Args:
        metricas (list): Lista de tuplas (nombre_subcarpeta, datos_json)
        
    Returns:
        dict: Diccionario con {nombre_subcarpeta: {'steps': [], 'values': []}}
    """
    datos_loss = {}
    metricaAExtraer = str(categoria + '.' +subcategoria )
    for nombre, ruta_metrics ,datos in metricas:
        if metricaAExtraer in datos:
            loss_data = datos[metricaAExtraer]
            datos_loss[nombre] = {
                'steps': loss_data['steps'],
                'values': loss_data['values']
            }
        else:
            print(f"Advertencia: No se encontró {metricaAExtraer} en {nombre}")
    
    return datos_loss

def graficar_loss(datos,nombreDeMetrica ):
    """
    Grafica las curvas de pérdida y opcionalmente guarda el gráfico
    
    Args:
        datos_loss (dict): Datos de pérdida por subcarpeta
        output_path (str): Opcional. Ruta para guardar el gráfico
    """
    if not datos:
        print("No hay datos de loss para graficar")
        return
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(datos)))
    output_path = os.getcwd()
    print(output_path)

    for i, (nombre, data) in enumerate(datos.items()):
        steps = data['steps']
        values = data['values']
        min_len = min(len(steps), len(values))
        
        plt.plot(steps[:min_len], values[:min_len], 
                label=nombre, 
                color=colors[i],
                linewidth=2,
                alpha=0.8)
    plt.yscale('log')
    plt.title(f'Comparación de {nombreDeMetrica}', fontsize=16)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    nombreDeMetrica = f"{nombreDeMetrica.replace('.', '_')}.png"
    output_path = os.path.join(output_path, nombreDeMetrica)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado como: {nombreDeMetrica}")
    
    plt.show()



metricas = obtener_metricas('../../test_runs')

subcategorias = ['loss.mean','loss.autoencoder.mean','loss.topo_error.mean','metrics.loss.mean','metrics.matched_pairs_0D.mean','reconstruction_error.mean']

for subcategoria in subcategorias:

    datos = extraer_datos(metricas,'training',subcategoria)
    nombremetrica = 'training' + '.' + subcategoria 
    graficar_loss(datos,nombremetrica)

subcategorias = ['loss','loss.autoencoder','loss.topo_error','metrics.loss','metrics.matched_pairs_0D','reconstruction_error']
for subcategoria in subcategorias:

    datos = extraer_datos(metricas,'validation',subcategoria)
    nombremetrica = 'validation' + '.' + subcategoria 
    graficar_loss(datos,nombremetrica)    
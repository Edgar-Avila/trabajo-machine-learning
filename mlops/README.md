# MLOps & Arquitectura de Automatización

Este módulo contiene la infraestructura de Integración Continua (CI), Despliegue Continuo (CD) y mantenimiento automático del modelo de Machine Learning.

## Arquitectura del Ciclo de Vida

```text
Código / Push ──► GitHub Actions (Lint + Pytest) ──► SSH Deploy ──► VPS (Docker Compose)
                                                                           ▲
Nuevos Datos  ──► Reentrenamiento ──► Gatekeeper (promote_model.py) ───────┘
                                       ├── ¿Supera métrica anterior? -> Promueve (current.txt)
                                       └── ¿Métrica inferior?        -> Rechaza (Protege Prod)
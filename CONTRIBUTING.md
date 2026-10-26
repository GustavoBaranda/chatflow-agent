# Contributing to chatflow-agent / Guía de Contribución

[English](#english) | [Español](#español)

---

## English

Thank you for your interest in contributing to **chatflow-agent**! We welcome contributions from developers of all experience levels. Whether you are fixing a typo in documentation, reporting an edge case, adding unit tests, or proposing a new channel connector, your help makes this project better for everyone.

### Code of Conduct

All contributors and maintainers are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating in discussions or opening pull requests.

---

### Getting Started

#### Prerequisites

- **Python 3.10, 3.11, 3.12, or 3.13** installed.
- **Git** installed and configured.
- (Optional) An API key for Google Gemini, OpenAI, Anthropic, or a local [Ollama](https://ollama.com/) instance if you want to run live LLM examples. Note that **running the unit test suite does not require any paid API keys or network access**.

#### 1. Fork and Clone the Repository

1. Fork the repository on GitHub by clicking the **Fork** button at [github.com/GustavoBaranda/chatflow-agent](https://github.com/GustavoBaranda/chatflow-agent).
2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/chatflow-agent.git
cd chatflow-agent
```

3. Add the upstream repository as a remote:

```bash
git remote add upstream https://github.com/GustavoBaranda/chatflow-agent.git
git fetch upstream
```

#### 2. Set Up a Virtual Environment

Create and activate an isolated virtual environment:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### 3. Install Development Dependencies

Install `chatflow-agent` in editable mode with all optional dependencies and development tooling:

```bash
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

This installs:
- Core dependencies: `pydantic`, `httpx`
- Channel connectors (`telegram`, `whatsapp`, `webhook`): `fastapi`, `uvicorn`, `python-telegram-bot`
- Developer & QA tools (`dev`): `pytest`, `pytest-asyncio`, `ruff`, `mypy`

#### 4. Verify Your Setup by Running Tests

Run the test suite to confirm that your environment is properly configured:

```bash
pytest
```

All unit and integration tests (1,200+ across the suite) should pass cleanly.

---

### Development Workflow

#### Branching Conventions

Always create a new branch from `upstream/main` for your work. Keep branches focused on a single change:

```bash
git checkout main
git pull upstream main
git checkout -b <type>/<short-description>
```

Recommended prefixes:
- `feat/`: New features, providers, or channel adapters
- `fix/`: Bug fixes and resilience patches
- `docs/`: Documentation, README, guides, or docstring improvements
- `test/`: Adding or improving tests
- `refactor/`: Code reorganization without functional changes
- `chore/`: Build configuration, CI workflows, or tooling updates

*Examples:* `fix/sqlite-session-lock`, `docs/clarify-async-tool-execution`, `feat/discord-channel`

#### Code Style & Quality Standards

`chatflow-agent` strictly enforces modern, clean, and typed Python standards:

##### 1. Linting & Formatting with Ruff

We use [Ruff](https://astral.sh/ruff) for linting and code formatting (`target-version = "py310"`).

Run the linter check:
```bash
ruff check .
```

To automatically apply safe fixes:
```bash
ruff check . --fix
```

##### 2. Static Type Checking with Mypy

We use [Mypy](https://mypy-lang.org/) in strict mode (`strict = true`). All new functions, methods, and classes must include explicit type annotations:

```bash
mypy
```

Ensure no type errors are raised before committing.

##### 3. Running Unit and Integration Tests

Run the complete test suite with verbose output:

```bash
pytest -v
```

To run a specific test file:
```bash
pytest tests/test_runner.py -v
```

To run tests matching a specific expression:
```bash
pytest -k "test_runner_with_tool" -v
```

#### Writing Tests

- Any bug fix should include a test demonstrating the issue and verifying the fix.
- Any new feature or engine should include unit tests with deterministic mocks (see `MockEngine` in `tests/test_runner.py` for inspiration).
- Do not make external HTTP network calls in unit tests; use mocks or fake engines.

#### Backward Compatibility

`chatflow-agent` is designed to be lightweight and easy to integrate:
- Do not introduce breaking changes to the public API exported in `src/chatflow_agent/__init__.py` (such as `Agent`, `Runner`, `Tool`, `tool`, engines, session stores/contexts, channels, domain types, and exceptions) unless explicitly agreed upon in a GitHub Issue.
- Avoid introducing heavy third-party dependencies into the core package (`dependencies` in `pyproject.toml`). If a feature requires extra libraries, make it an optional dependency under `[project.optional-dependencies]`.

---

### Pull Request (PR) Guidelines

When you are ready to submit your changes:

1. **Keep PRs atomic and focused:** A PR should solve one problem or add one feature. Smaller PRs are reviewed and merged much faster.
2. **Push your branch to your fork:**
   ```bash
   git push origin <type>/<short-description>
   ```
3. **Open a Pull Request:** Navigate to [github.com/GustavoBaranda/chatflow-agent](https://github.com/GustavoBaranda/chatflow-agent) and click **New Pull Request**.
4. **Fill out the Pull Request Template:**
   - Link the relevant issue (e.g., `Fixes #42` or `Closes #18`).
   - Describe the motivation and changes made.
   - List the tests executed.
   - Complete the review checklist.
5. **Verify CI Status:** Ensure all GitHub Actions workflow checks pass (Python 3.10-3.13 on Ubuntu and Windows).
6. **Address Review Feedback:** Maintainers may suggest improvements. You can push additional commits to the same branch, and the PR will update automatically.

---

### Community Guidelines

#### How to Report Bugs

If you find a bug:
1. Search [existing GitHub Issues](https://github.com/GustavoBaranda/chatflow-agent/issues) to verify it hasn't already been reported.
2. Open a new issue using the **Bug Report** template.
3. Provide a clear description, minimal reproducible Python code, your OS, and Python version.
4. **Security Notice:** If you have discovered a security vulnerability, do NOT open a public issue. Follow our [Security Policy](SECURITY.md). Never include raw API keys, bearer tokens, passwords, or personal telephone numbers in issue reports or logs.

#### How to Propose Enhancements

1. Open a new issue using the **Feature Request** template.
2. Describe the problem, the proposed solution, and realistic use cases.
3. Review the [Architecture Overview](docs/ARCHITECTURE.md) and discuss the architectural fit before submitting a large pull request to avoid rework.

#### Finding Good First Issues

Looking for a place to start? Check the [Issue Tracker](https://github.com/GustavoBaranda/chatflow-agent/issues) filtered by:
- `good first issue`: Self-contained tasks ideal for newcomers to the codebase.
- `help wanted`: Tasks where the community is actively seeking contributions.
- `documentation`: Improvements to guides, docstrings, or examples.

---

## Español

¡Gracias por tu interés en contribuir a **chatflow-agent**! Damos la bienvenida a colaboraciones de desarrolladores de todos los niveles de experiencia. Ya sea corrigiendo un error tipográfico en la documentación, reportando un caso borde, agregando pruebas unitarias o proponiendo un nuevo conector de canal, tu ayuda hace que este proyecto sea mejor para todos.

### Código de Conducta

Se espera que todos los colaboradores y mantenedores sigan nuestro [Código de Conducta](CODE_OF_CONDUCT.md). Por favor, léelo antes de participar en discusiones o abrir pull requests.

---

### Primeros Pasos

#### Requisitos Previos

- **Python 3.10, 3.11, 3.12 o 3.13** instalado.
- **Git** instalado y configurado.
- (Opcional) Una API key de Google Gemini, OpenAI, Anthropic o una instancia local de [Ollama](https://ollama.com/) si deseas probar ejemplos con LLMs en vivo. **La ejecución de la suite de pruebas no requiere API keys pagas ni acceso a internet**.

#### 1. Bifurcar (Fork) y Clonar el Repositorio

1. Haz un fork del repositorio en GitHub haciendo clic en el botón **Fork** en [github.com/GustavoBaranda/chatflow-agent](https://github.com/GustavoBaranda/chatflow-agent).
2. Clona tu fork localmente:

```bash
git clone https://github.com/<tu-usuario>/chatflow-agent.git
cd chatflow-agent
```

3. Agrega el repositorio upstream como remoto:

```bash
git remote add upstream https://github.com/GustavoBaranda/chatflow-agent.git
git fetch upstream
```

#### 2. Configurar el Entorno Virtual

Crea y activa un entorno virtual aislado:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Símbolo del Sistema / CMD):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### 3. Instalar Dependencias de Desarrollo

Instala `chatflow-agent` en modo editable con todas las dependencias opcionales y herramientas de desarrollo:

```bash
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

Esto instala:
- Dependencias del núcleo: `pydantic`, `httpx`
- Conectores de canales (`telegram`, `whatsapp`, `webhook`): `fastapi`, `uvicorn`, `python-telegram-bot`
- Herramientas de desarrollo y testing (`dev`): `pytest`, `pytest-asyncio`, `ruff`, `mypy`

#### 4. Verificar la Instalación Ejecutando las Pruebas

Ejecuta la suite de pruebas para confirmar que tu entorno está correctamente configurado:

```bash
pytest
```

Todas las pruebas unitarias e integrales (más de 1.200 en la suite) deben pasar exitosamente.

---

### Flujo de Desarrollo

#### Convenciones de Ramas

Crea siempre una nueva rama a partir de `upstream/main` para tu trabajo. Mantén las ramas enfocadas en un único cambio:

```bash
git checkout main
git pull upstream main
git checkout -b <tipo>/<descripcion-corta>
```

Prefijos recomendados:
- `feat/`: Nuevas funcionalidades, proveedores o adaptadores de canales
- `fix/`: Correcciones de errores y parches de resiliencia
- `docs/`: Mejoras en documentación, README, guías o docstrings
- `test/`: Incorporación o mejora de pruebas
- `refactor/`: Reorganización de código sin cambios funcionales
- `chore/`: Configuración de build, flujos de CI o herramientas

*Ejemplos:* `fix/sqlite-session-lock`, `docs/clarify-async-tool-execution`, `feat/discord-channel`

#### Estándares de Calidad y Estilo de Código

`chatflow-agent` aplica estándares estrictos de Python moderno y tipado estático:

##### 1. Linting y Formato con Ruff

Utilizamos [Ruff](https://astral.sh/ruff) para linting y formateo de código (`target-version = "py310"`).

Ejecutar la verificación del linter:
```bash
ruff check .
```

Para aplicar correcciones automáticas seguras:
```bash
ruff check . --fix
```

##### 2. Chequeo de Tipos Estático con Mypy

Utilizamos [Mypy](https://mypy-lang.org/) en modo estricto (`strict = true`). Todas las nuevas funciones, métodos y clases deben incluir anotaciones de tipo explícitas:

```bash
mypy
```

Asegúrate de que no se reporte ningún error de tipado antes de commitear.

##### 3. Ejecución de Pruebas con Pytest

Ejecutar la suite completa con salida detallada:

```bash
pytest -v
```

Para ejecutar un archivo de pruebas específico:
```bash
pytest tests/test_runner.py -v
```

Para ejecutar pruebas que coincidan con una expresión:
```bash
pytest -k "test_runner_with_tool" -v
```

#### Escritura de Pruebas

- Toda corrección de error debe incluir una prueba que reproduzca el problema y verifique la solución.
- Toda nueva funcionalidad debe incluir pruebas unitarias con mocks deterministas (consulta `MockEngine` en `tests/test_runner.py` como referencia).
- No realices llamadas HTTP de red externas en las pruebas unitarias; utiliza mocks o motores simulados.

#### Compatibilidad Hacia Atrás

`chatflow-agent` está diseñado para ser liviano y fácil de integrar:
- No introduzcas cambios que rompan la API pública exportada en `src/chatflow_agent/__init__.py` (como `Agent`, `Runner`, `Tool`, `tool`, motores, tiendas/contextos de sesión, canales, tipos de dominio y excepciones) salvo acuerdo previo en un Issue de GitHub.
- Evita introducir dependencias pesadas de terceros en el paquete base (`dependencies` en `pyproject.toml`). Si una funcionalidad requiere librerías adicionales, declárala como dependencia opcional en `[project.optional-dependencies]`.

---

### Guía para Pull Requests (PR)

Cuando tengas listos tus cambios:

1. **Mantén los PRs atómicos y enfocados:** Un PR debe resolver un único problema o agregar una única funcionalidad. Los PRs pequeños se revisan y mergean mucho más rápido.
2. **Sube tu rama a tu fork:**
   ```bash
   git push origin <tipo>/<descripcion-corta>
   ```
3. **Abre un Pull Request:** Ve a [github.com/GustavoBaranda/chatflow-agent](https://github.com/GustavoBaranda/chatflow-agent) y haz clic en **New Pull Request**.
4. **Completa la Plantilla de Pull Request:**
   - Vincula el issue relacionado (por ejemplo, `Fixes #42` o `Closes #18`).
   - Describe la motivación y los cambios realizados.
   - Lista las pruebas ejecutadas.
   - Completa la lista de verificación (checklist).
5. **Verifica el Estado del CI:** Asegúrate de que todos los checks de GitHub Actions pasen (Python 3.10-3.13 en Ubuntu y Windows).
6. **Responde a los Comentarios de Revisión:** Los mantenedores pueden sugerir ajustes. Puedes subir commits adicionales a la misma rama y el PR se actualizará automáticamente.

---

### Guías Comunitarias

#### Cómo Reportar Errores

Si encuentras un error:
1. Busca en los [Issues existentes en GitHub](https://github.com/GustavoBaranda/chatflow-agent/issues) para verificar que no haya sido reportado previamente.
2. Abre un nuevo issue usando la plantilla de **Bug Report**.
3. Proporciona una descripción clara, código mínimo reproducible en Python, tu sistema operativo y la versión de Python.
4. **Aviso de Seguridad:** Si has descubierto una vulnerabilidad de seguridad, ¡NO abras un issue público! Sigue nuestra [Política de Seguridad](SECURITY.md). Nunca incluyas API keys reales, tokens de acceso, contraseñas ni números telefónicos privados en reportes o logs.

#### Cómo Proponer Mejoras

1. Abre un nuevo issue usando la plantilla de **Feature Request**.
2. Describe el problema, la solución propuesta y casos de uso del mundo real.
3. Consulta la [Descripción de Arquitectura](docs/es/ARCHITECTURE.md) y debate el encaje arquitectónico antes de enviar un PR de gran tamaño para evitar retrabajos.

#### Cómo Elegir un Issue para Comenzar

¿Buscás por dónde empezar? Revisa el [Issue Tracker](https://github.com/GustavoBaranda/chatflow-agent/issues) filtrando por:
- `good first issue`: Tareas autocontenidas ideales para quienes se inician en el código.
- `help wanted`: Tareas donde la comunidad busca activamente colaboradores.
- `documentation`: Mejoras en guías, docstrings o ejemplos.

---

¡Gracias por contribuir a hacer de `chatflow-agent` un framework multiagente accesible y robusto!

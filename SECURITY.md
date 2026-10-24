# Security Policy / Política de Seguridad

[English](#english) | [Español](#español)

---

## English

### Supported Versions

Security patches and bug fixes are provided exclusively for the latest released version on a best-effort basis. Prior versions are not supported and do not receive maintenance or backported security updates.

| Version | Supported | Notes |
|:-------:|:---------:|:------|
| `0.3.4` (Latest) | :white_check_mark: | Current release supported on a best-effort basis. |
| `< 0.3.4` | :x: | Unsupported. Please upgrade to the latest release (`0.3.4`). |

---

### Reporting a Vulnerability

We take the security of `chatflow-agent` seriously. If you discover a security vulnerability, **never disclose it publicly in GitHub Issues, pull requests, discussions, or social media**. Public issues are not a secure channel for reporting vulnerabilities.

#### Preferred Method: GitHub Private Vulnerability Reporting

Whenever enabled on this repository, the safest and preferred method to report a vulnerability is directly through GitHub Private Vulnerability Reporting:

1. Navigate to the repository: [github.com/GustavoBaranda/chatflow-agent](https://github.com/GustavoBaranda/chatflow-agent).
2. Click on the **Security** tab.
3. Select **Advisories** on the left menu (or visit [Security Advisories](https://github.com/GustavoBaranda/chatflow-agent/security/advisories)).
4. Click **Report a vulnerability** to submit a private draft advisory directly to the maintainer.

#### Fallback Method: Contact via GitHub Profile

If GitHub Private Vulnerability Reporting is unavailable or disabled, you may reach out to the project maintainer privately through their official GitHub profile:
[https://github.com/GustavoBaranda](https://github.com/GustavoBaranda)

> [!CAUTION]
> Never open a public issue to report security vulnerabilities or sensitive security findings.

---

### What to Include in a Report

To help triage and resolve the issue effectively, please include as much detail as possible:

- **Type of vulnerability** (e.g., webhook bypass, credential exposure, denial of service, memory leak, unvalidated input).
- **Affected component(s)** (e.g., `WhatsAppChannel`, `OpenAIEngine`, `Runner`, `SQLiteSessionStore`).
- **Steps to reproduce** or a minimal proof-of-concept (PoC) script.
- **Potential security impact** and how an attacker could exploit the issue.
- **Any suggested mitigations** or patches, if available.

> [!WARNING]
> Please do not include production secrets, live Meta access tokens, or private phone numbers in your report. Always use sanitized mock values.

---

### Response and Disclosure Process

`chatflow-agent` is an independent open-source project maintained primarily by its author. Security reports are handled on a best-effort basis according to maintainer availability without fixed SLAs:

1. **Acknowledgment:** We strive to acknowledge receipt of your report as soon as practicable.
2. **Assessment:** We will confirm the vulnerability, assess its impact, and coordinate remediation.
3. **Patch & Release:** Once verified, a fix will be published in a new release on PyPI, accompanied by a GitHub Security Advisory crediting the reporter (unless anonymity is requested).
4. **Public Disclosure:** Public disclosure takes place only after the fix has been released, allowing downstream users time to upgrade.

---

## Español

### Versiones con Soporte

Los parches de seguridad y correcciones de errores se proporcionan exclusivamente para la última versión publicada bajo la modalidad de mejor esfuerzo (*best-effort*). Las versiones anteriores no cuentan con soporte ni reciben mantenimiento o parches retrocompatibles.

| Versión | Con Soporte | Notas |
|:-------:|:-----------:|:------|
| `0.3.4` (Última) | :white_check_mark: | Versión actual soportada bajo modalidad best-effort. |
| `< 0.3.4` | :x: | Sin soporte. Actualiza a la última versión disponible (`0.3.4`). |

---

### Cómo Reportar una Vulnerabilidad

Nos tomamos muy en serio la seguridad de `chatflow-agent`. Si descubres una vulnerabilidad de seguridad, **nunca la divulgues públicamente en GitHub Issues, pull requests, discusiones ni redes sociales**. Los issues públicos no son un canal seguro para reportar vulnerabilidades.

#### Método Preferido: Reporte Privado de Vulnerabilidades de GitHub

Siempre que la funcionalidad se encuentre habilitada en el repositorio, la vía recomendada y más segura para reportar vulnerabilidades es a través del sistema de Reporte Privado de Vulnerabilidades de GitHub (*GitHub Private Vulnerability Reporting*):

1. Ingresa al repositorio: [github.com/GustavoBaranda/chatflow-agent](https://github.com/GustavoBaranda/chatflow-agent).
2. Haz clic en la pestaña **Security** (Seguridad).
3. Selecciona **Advisories** (Avisos de seguridad) en el menú lateral izquierdo (o visita [Avisos de Seguridad](https://github.com/GustavoBaranda/chatflow-agent/security/advisories)).
4. Haz clic en **Report a vulnerability** para enviar un borrador privado directamente al mantenedor.

#### Método Alternativo: Contacto a través del Perfil de GitHub

Si el reporte privado de vulnerabilidades de GitHub no estuviera disponible o habilitado, puedes ponerte en contacto de forma privada con el mantenedor a través de su perfil oficial de GitHub:
[https://github.com/GustavoBaranda](https://github.com/GustavoBaranda)

> [!CAUTION]
> Nunca abras un issue público para reportar vulnerabilidades de seguridad ni hallazgos sensibles.

---

### Qué Incluir en el Reporte

Para facilitar la evaluación y resolución del problema, incluye tantos detalles como sea posible:

- **Tipo de vulnerabilidad** (por ejemplo: evasión de firmas en webhooks, exposición de credenciales, denegación de servicio, fuga de memoria, entrada no validada).
- **Componente(s) afectado(s)** (por ejemplo: `WhatsAppChannel`, `OpenAIEngine`, `Runner`, `SQLiteSessionStore`).
- **Pasos para reproducir** o un script mínimo de prueba de concepto (PoC).
- **Impacto potencial de seguridad** y cómo un atacante podría explotar la debilidad.
- **Mitigaciones o parches sugeridos**, si los tienes disponibles.

> [!WARNING]
> Nunca incluyas secretos de producción, tokens reales de acceso de Meta ni números telefónicos privados en tu reporte. Utiliza siempre valores ficticios sanitizados.

---

### Proceso de Respuesta y Divulgación

`chatflow-agent` es un proyecto open source independiente desarrollado y mantenido principalmente por su autor. Los reportes se gestionan bajo la modalidad de mejor esfuerzo (*best-effort*) según la disponibilidad del mantenedor, sin acuerdos de nivel de servicio (SLA) fijados:

1. **Acuse de recibo:** Procuramos responder confirmando la recepción del reporte tan pronto como resulte factible.
2. **Evaluación:** Confirmaremos la vulnerabilidad, determinaremos su severidad y coordinaremos las medidas correctivas.
3. **Parche y Publicación:** Una vez verificado el parche, se publicará una nueva versión en PyPI, acompañada de un aviso de seguridad en GitHub acreditando a quien lo reportó (salvo que solicite anonimato).
4. **Divulgación Pública:** La divulgación pública se realizará únicamente después de que el parche esté disponible, dando tiempo a los usuarios para actualizar sus entornos.

¡Gracias por ayudar a mantener seguro a `chatflow-agent` y a su comunidad!

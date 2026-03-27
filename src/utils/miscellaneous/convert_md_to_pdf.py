import pypandoc
import re

if __name__ == "__main__":
  input_file = r"C:\workspace\hd-image-pest-detection-cnn-and-rf\BITACORA.md"
  output_file = r"BITACORA.pdf"

  with open("BITACORA.md", "r", encoding="utf-8") as f:
      content = f.read()

  # Eliminar caracteres tipo ━
  content = content.replace("━", "-")
  content = re.sub(r'[^\x00-\x7F]+', '', content)


  with open("BITACORA_clean.md", "w", encoding="utf-8") as f:
      f.write(content)

  pypandoc.convert_file(
      input_file,
      'pdf',
      outputfile=output_file,
      extra_args=[
        '--standalone',
        '--pdf-engine=xelatex',
        '--listings', 
        # Márgenes
        '-V', 'geometry:margin=1in',

        # Ajuste de texto automático
        '-V', 'fontsize=11pt',

        # Mejor manejo de líneas largas
        '-V', 'linestretch=1.2',

        # Fuente moderna (mejora render)
        '-V', 'mainfont=Arial',

        # Permite cortar palabras largas
        '-V', 'papersize=A4',

        # 🔥 CLAVE: habilita wrap en código
        '--wrap=auto',

        '--highlight-style=pygments',
        ],
  )

  print("✅ PDF generado correctamente")
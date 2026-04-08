def tratamento_nomes(nomes_txt):
  with open(f"{nomes_txt}", "r", encoding="utf-8") as f:
      nomes = f.readlines()

  nomes_formatados = [nome.strip().lower() for nome in nomes]

  with open("nomes_formatados.txt", "w", encoding="utf-8") as f:
      f.write("\n".join(nomes_formatados))

tratamento_nomes("nomes.txt")
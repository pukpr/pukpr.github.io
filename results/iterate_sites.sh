#!/bin/bash

# Usage: ./process_csv_first_field.sh input.csv your_program
# Example: ./process_csv_first_field.sh data.csv echo

CSV_FILE="$1"
PROGRAM="./tmodel.sh"
touch metrics.html
rm metrics.html
echo "<pre>" >  metrics.html

if [[ -z "$CSV_FILE" || -z "$PROGRAM" ]]; then
  echo "Usage: $0 input.csv program_to_run"
  exit 1
fi

# Read each line, extract the first field, and call the program
while IFS=, read -r first_field _; do
  # Skip empty lines
  [[ -z "$first_field" ]] && continue
  #env EXTRA=0.408 ./tmodel.sh "$first_field" true -0.01 0.1 0
  env EXTRA=0.408 ./tmodel.sh "$first_field" $2 $3 $4 0
  echo "$first_field" >> metrics.html
  cat metrics.txt >> metrics.html
done < "$CSV_FILE"

echo "</pre>" >>  metrics.html

mv ./*site$3-$4.png ~/github/pukpr/pukpr.github.io/results/$5
mv ./metrics.html   ~/github/pukpr/pukpr.github.io/results/$5

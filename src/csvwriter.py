import csv
import re

# Read the content from the file
def csvwriter(iter=None):
    with open('log1.txt', 'r') as file:
        content = file.read()

    # Extract macro information
    macro_pattern = re.compile(r'Macro Number:(\d+), width:(\d+),height:(\d+)')
    macros = macro_pattern.findall(content)

    # Extract netlist information
    netlist_pattern = re.compile(r'Net n\d+ connected between m(\d+) and m(\d+)')
    netlist = netlist_pattern.findall(content)

    # Extract HPWL values
    hpwl_pattern = re.compile(r'HPWL: ([\d.]+)')
    hpwl_values = hpwl_pattern.findall(content)

    # Prepare data for CSV
    macros_data = [(int(width),int(height)) for num, width, height in macros]
    netlist_data = [(int(m1),int(m2)) for i, (m1, m2) in enumerate(netlist)]
    hpwl_data = [value for i, value in enumerate(hpwl_values)]

    # Write to CSV
    if iter==0: csvfile = open('output.csv', 'w', newline='')
    else: csvfile = open('output.csv', 'a', newline='')
    csvwriter = csv.writer(csvfile)
    if iter==0: csvwriter.writerow(['Macros', 'Netlist', 'HPWL'])

    # Combine all data into a single row
    csvwriter.writerow([
        list(macros_data),
        list(netlist_data),
        hpwl_data
    ])

    csvfile.close()

# Call the function to execute the CSV writing
csvwriter()
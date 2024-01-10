def matrix_print(a, format_str="{:8.3f}"):
    for row in a:
        for col in row:
            print(format_str.format(col), end=" ")
        print("")
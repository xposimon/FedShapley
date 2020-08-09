with open("My_MRnoise12half.txt", "r") as f:
    content = f.readlines()

contr = []

for i,line in enumerate(content):
    if line.startswith("round") and line[6] != '0':
        tmp = content[i-2][1:-2].split(",")
        tmp = [float(k.strip()) for k in tmp]
        contr.append(tmp)
print(len(contr))

for i in range(3):
    # with open("MRsame_round_contr%d.out"%(i), "w") as f:
    #     for j in range(len(contr)):
    #         if j >= 1:
    #             f.write("(%d, %f)\n"%(j+1, contr[j][i]-contr[j-1][i]))
    #         else:
    #             f.write("(%d, %f)\n"%(j+1, contr[j][i]))
    with open("MRnoise12_accu_contr%d.out"%(i), "w") as f:
        for j in range(len(contr)):
            f.write("(%d, %f)\n"%(j+1, contr[j][i]))
            


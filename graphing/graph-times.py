from matplotlib import pyplot as plt


def makeWaveformPlot():
    #the inputs are text files
    times = []
    freq_model = []
    mass_model = []
    mass_model2 = []
    mass_model3 = []

    f = open("times.txt",'r') 
    freqfile = open("AET_FTs_freqmodel.txt", 'r')
    massfile = open("AET_FTs_massmodelv2.txt", 'r')
    massfile2 = open("AET_FTs_massmodel_adjustedm1,m2.txt", 'r')
    massfile3 = open("AET_FTs_massmodel_m1m2.txt", 'r')
    for row in f: 
        row = row.split('\n') 
        times.append(float(row[0]))  
    for row in freqfile: 
        row = row.split('\n') 
        freq_model.append(float(row[0])) 
    for row in massfile: 
        row = row.split('\n') 
        mass_model.append(float(row[0]))   
    for row in massfile2: 
        row = row.split('\n') 
        mass_model2.append(float(row[0]))   
    for row in massfile3: 
        row = row.split('\n') 
        mass_model3.append(float(row[0]))   

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    ax1.plot(times, freq_model, label="Freq model", color="blue")
    ax2.plot(times, mass_model, label="Mc, Iwd model", color="orange") 
    ax2.plot(times, mass_model2, label="Model with adjusted m1, m2", color="green")
    ax2.plot(times, mass_model3, label="Model with OG m1, m2", color="red")  
    plt.title("AET_FTs vs TTs for Models")
    plt.xlabel("TTs")
    plt.legend()
    ax1.set_ylabel("Freq Model AET_FTs")
    ax2.set_ylabel("Mass Model AET_FTs")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
makeWaveformPlot()
import multiprocessing
import time
import random
from selenium import webdriver
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders
import numpy
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import cv2
import glob, os
import sys

def init_window():
    pygame.init()
    pygame.display.set_mode((1000, 1000), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption('testi')
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_CLIP_DISTANCE0)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

def loading_shaiba(loading_model):
    vertex_shader = """
                        #version 330
                        in vec3 position;
                        in vec3 color;

                        out vec3 newColor;

                        void main()
                        {
                            gl_Position = vec4(position,1.0f);
                            newColor = color;
                        }
                        """

    fragment_shader = """
                        #version 330
                        in vec3 newColor;


                        out vec4 outColor;
                        void main()
                        {
                            outColor = vec4(newColor, 1.0f);
                            //outcoloreilla syvyyskayrat
                            //if ((outColor.x > 0.595) && (outColor.x < 0.60)){
                                        //outColor = vec4(1, 0, 0, 1);
                            //}
                        }
                        """

    loading_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))

    #loading_model = [-1, -1, 0.0, 1.0, 1.0, 1.0,
    #                1, -1, 0.0, 0.5, 0.5, 0.5,
     #               0.0, 1, 0.0, 0.0, 0.0, 0.0]

    #loading_model = [0.72043269230769225, 0.46107692307692305, 0.0, 0.21052631578947367, 0.21052631578947367, 0.21052631578947367,
     #0.97978846153846144, 0.52831730769230767, 0.0, 0.9210526315789473, 0.9210526315789473, 0.9210526315789473,
     #0.26896153846153847, 0.45147115384615383, 0.0, 0.18421052631578946, 0.18421052631578946, 0.18421052631578946]

    loading_model = numpy.array(loading_model, dtype=numpy.float32)

    byytit = (len(loading_model))*4

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, byytit, loading_model, GL_STATIC_DRAW)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    position = glGetAttribLocation(loading_shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(loading_shader, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    return loading_shader,VAO

def draw_loading(loader_shader,VBO):
    glUseProgram(loader_shader[0])
    glClearColor(0,0,0,1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindVertexArray(loader_shader[1])
    glDrawArrays(GL_TRIANGLES,0,int(len(VBO)/6))
    glUseProgram(0)
    glBindVertexArray(0)

def skaalaus(syvyydet, koordit):
    minimi = min(syvyydet)
    maksimi = max(syvyydet)

    xkord = []
    ykord = []
    for z in koordit:
        xkord.append(z[1])
        ykord.append(z[0])

    xmax = max(xkord)
    xmin = min(xkord)
    ymax = max(ykord)
    ymin = min(ykord)

    Lx = xmax - xmin
    Ly = ymax - ymin


    if Lx >= Ly:
        kscaler = 0.999 / Lx
    else:
        kscaler = 0.999 / Ly

    for lel in range(len(xkord)):
        xkord[lel] = ((xkord[lel] - xmin) * kscaler)*2 -1
        ykord[lel] = ((ykord[lel] - ymin) * kscaler)*2 -1
        koordit[lel] = (xkord[lel], ykord[lel])

    for j in range(len(syvyydet)):
        syvyydet[j] = syvyydet[j] - minimi

    kerroin = 1.0 / maksimi

    for i in range(len(syvyydet)):
        syvyydet[i] = float(syvyydet[i] * kerroin)

    kjas = []
    for x in range(len(koordit)):
        kjas.append((koordit[x], syvyydet[x]))

    return kjas, kerroin, maksimi, minimi, xmin, ymin, kscaler

def veneile(naytto):
    screen = pygame.display.set_mode(naytto)
    mouseposlist = []
    syvyyslist = []
    tausta = pygame.image.load("Heightmap.png")
    pygame.Surface.blit(screen, tausta, (0, 0))
    pygame.display.flip()

    tartti = False
    toppi = False

    counter = 0
    while not toppi:  # len(mouseposlist) < 1000: #veneilymatka
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                tartti = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_e:
                toppi = True

        time.sleep(0.01)

        if tartti:
            mopo = pygame.mouse.get_pos()
            syv = pygame.Surface.get_at(screen, mopo)
            syvyyslist.append(syv[0])
            mouseposlist.append(mopo)
            counter += 1
            print(counter)
            pygame.display.update()

    pygame.quit()
    return mouseposlist, syvyyslist

def veneile_real_geo():
    on_mapping = True
    geo_cord_lista = []
    syvyyslista = []

    counter = 0
    while on_mapping:
        #inppi = input("q to stop Syvyys?: ")
        time.sleep(0.2)
        inppi = random.randint(0,10)
        #if inppi == "q":
        if counter == 10:
            on_mapping = False
        else:
            try:
                syvyyslista.append(float(inppi))
                lantti = lat_multi.value
                lontti = lon_multi.value
                geo_cord_lista.append([lantti,lontti])
                print([lantti,lontti])
            except:
                print("paska_value")
                pass
        counter += 1

    return geo_cord_lista, syvyyslista

def geo2kord(geokoordit):
    lats = []
    lons = []
    for i in range(len(geokoordit)):
        lats.append(geokoordit[i][0])
        lons.append(geokoordit[i][1])

    latmin = min(lats)
    lonmin = min(lons)
    # kuvan reuna koordinaatistossa

    suhdeluku = ((111.320 * numpy.cos(numpy.deg2rad(latmin))) / 110.574)  # lon koordin pituus verrattuna lattiin

    for z in range(len(lats)):
        lats[z] = lats[z] - latmin
        lons[z] = ((lons[z] - lonmin) * suhdeluku)

    taapalautetaan = []
    for nig in range (len(lats)):
        taapalautetaan.append([lats[nig],lons[nig]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lons, lats, 'r')
    ax.set_aspect('equal')
    ax.grid()
    plt.show()

    return taapalautetaan, suhdeluku, latmin, lonmin

def kappura_arvot(skaalattu):
    kerroin = skaalattu[1]
    maksimi = skaalattu[2]
    minimi = skaalattu[3]
    vaihteluvali = maksimi - minimi
    syv_1 = str(round((20 / 255) * vaihteluvali + minimi))
    syv_2 = str(round((50 / 255) * vaihteluvali + minimi))
    syv_3 = str(round((100 / 255) * vaihteluvali + minimi))
    syv_4 = str(round((150 / 255) * vaihteluvali + minimi))
    syv_5 = str(round((200 / 255) * vaihteluvali + minimi))

    return syv_1, syv_2, syv_3, syv_4, syv_5

def for_triags(pisteet_syvyys):
    just_test = []
    for a in pisteet_syvyys:
        just_test.append(a[0])
    #print(just_test)
    just_test = numpy.asarray(just_test)
    return just_test

def triaggel(just_test, pisteet_syvyys):
    xs = []
    ys = []
    for x, y in just_test:
        xs.append(x)
        ys.append(y)

    # plt.show()
    # print(np.asarray(triang))
    Dtri = Delaunay(just_test)
    jahas = Dtri.vertices  # kaytannossa planepointterit

    triaglist = []
    syvarilista = []
    VBO = []
    for o in jahas:
        triag = []
        syvari = []
        for t in o:
            triag.append(just_test[t][0])
            triag.append(just_test[t][1])
            triag.append(0.0)
            triag.append(pisteet_syvyys[t][1])
            triag.append(pisteet_syvyys[t][1])
            triag.append(pisteet_syvyys[t][1])
            #syvari.append(pisteet_syvyys[t][1])

        triaglist.append(triag)
        syvarilista.append(syvari)

    # print(triaglist)
    # print(syvarilista)
    plt.triplot(just_test[:, 0], just_test[:, 1], Dtri.simplices)
    plt.plot(just_test[:, 0], just_test[:, 1], 'o')
    plt.show()
    return triaglist, syvarilista

def VBOIze(triag_data):
    VBO = []
    for n in triag_data:
        for pala in n:
            VBO.append(pala)
    return VBO

def readScreen(x, y, width, height):
    """ Read in the screen information in the area specified """
    glFinish()
    glPixelStorei(GL_PACK_ALIGNMENT, 4)
    glPixelStorei(GL_PACK_ROW_LENGTH, 0)
    glPixelStorei(GL_PACK_SKIP_ROWS, 0)
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0)

    data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    if hasattr(data, "tostring"):
        data = data.tostring()

    return data

def kappyrat_kuvaan(kuva,karvot):
    pix = kuva.load()
    width, height = kuva.size

    for x in range(width):
        for y in range(height):
            if (pix[x, y][0] > 50) and (pix[x, y][0] < 52):
                pix[x, y] = 194, 5, 148, 255

            elif (pix[x, y][0] > 100) and (pix[x, y][0] < 102):
                pix[x, y] = 129, 4, 196, 255

            elif (pix[x, y][0] > 150) and (pix[x, y][0] < 152):
                pix[x, y] = 0, 0, 255, 255

            elif (pix[x, y][0] > 200) and (pix[x, y][0] < 202):
                pix[x, y] = 0, 200, 200, 255

            elif (pix[x, y][0] > 20) and (pix[x, y][0] < 22):
                pix[x, y] = 255, 0, 0, 255

                # for pikseli in pisteet_syvyys:
                # pix[pikseli[0][0], pikseli[0][1]] = 255, 0, 0 ,255

    txt = Image.new('RGBA', kuva.size, (255, 255, 255, 0))
    fnt = ImageFont.truetype("arial.ttf", 28)

    d = ImageDraw.Draw(txt)
    d.text((20, 20), karvot[0], font=fnt, fill=(255, 0, 0, 255))
    d = ImageDraw.Draw(txt)
    d.text((20, 50), karvot[1], font=fnt, fill=(194, 5, 148, 255))
    d = ImageDraw.Draw(txt)
    d.text((20, 80), karvot[2], font=fnt, fill=(129, 4, 196, 255))
    d = ImageDraw.Draw(txt)
    d.text((20, 110), karvot[3], font=fnt, fill=(0, 0, 255, 255))
    d = ImageDraw.Draw(txt)
    d.text((20, 140), karvot[4], font=fnt, fill=(0, 200, 200, 255))

    out = Image.alpha_composite(kuva, txt)

    return out

def veneilyt(suhdeluku, latmin, lonmin, kscaler, kuva):
    while True:
        lantti = lat_multi.value
        lontti = lon_multi.value

        lantti = (lantti - latmin)*(kscaler*1000)
        lontti = (((lontti - lonmin) * suhdeluku))*(kscaler*1000)
        tuple_lalo = (int(lontti),int(lantti))
        #print(lontti,lantti)

        if lantti > 0 and lontti > 0 and lantti < 1000 and lontti < 1000:
            img_np = numpy.array(kuva)
            cv2.circle(img_np, tuple_lalo, 20, (0, 0, 255), -1)
            cv2.imshow("ESC to close NOOB", img_np)
            if cv2.waitKey(1) == 27:
                break

#def loadmap():


def main(lat_multi, lon_multi, jono):


    jono.get()

    while True:
        inp = input("1 create map, 2 load map")

        if inp == "1":
            veneilyt_base = veneile_real_geo()
            veneily_krd = geo2kord(veneilyt_base[0])

            skaalattu = skaalaus(veneilyt_base[1], veneily_krd[0])
            pisteet_syvyys = skaalattu[0]
            triag_gen_data = for_triags(pisteet_syvyys)
            triag_syvari = triaggel(triag_gen_data, pisteet_syvyys)
            VBO = VBOIze(triag_syvari[0])

            init_window()
            loading_shader = loading_shaiba(VBO)

            draw_loading(loading_shader,VBO)
            data = readScreen(2, 2, 1000, 1000)
            surface = pygame.image.fromstring(data, (1000, 1000), 'RGB', 1)
            pygame.image.save(surface, "temp.jpg")
            pygame.quit()

            image = Image.open("temp.jpg").convert("RGBA")
            im_mirror = ImageOps.flip(image)
            im_blur = im_mirror.filter(ImageFilter.GaussianBlur(8))

            karvot = kappura_arvot(skaalattu)
            im_kappyra = kappyrat_kuvaan(im_blur,karvot)

            ok = True
            while ok:
                try:
                    mapname = input("New map name?: ")
                    mapnamepng = mapname + ".png"
                    mapnametxt = mapname + ".txt"


                    im_kappyra.save(mapnamepng)
                    with open(mapnametxt,"w") as f:
                        f.write(str(veneily_krd[1])+",")
                        f.write(str(veneily_krd[2])+",")
                        f.write(str(veneily_krd[3])+",")
                        f.write(str(skaalattu[6])+",")

                    ok = False
                except:
                    print("paska nimi")
                    pass

        elif inp == "2":
            print("SAVED files")
            os.chdir(sys.path[0])
            for file in glob.glob("*.txt"):
                print(file[:-4])

            ok = True
            while ok:
                try:
                    filetopen = input("File to open?: ")
                    filetopentxt = filetopen + ".txt"
                    filetopenong = filetopen + ".png"

                    with open(filetopentxt) as f:
                        file = f.read().split(",")
                        del file[-1]
                    for i in range(len(file)):
                        file[i] = float(file[i])
                    #print(file)


                    ok = False
                except:
                    print("paska nimi")
                    pass

            image = Image.open(filetopenong)
            img_np = numpy.array(image)
            veneilyt(file[0], file[1], file[2], file[3], img_np)

        else:
            print("nig")

    p2.terminate()

def arduino_simulaattori(lat_multi, lon_multi, jono):


    geo_list = []
    geo_raw = []
    with open("krdkeuruu.txt") as f:
        for line in f:
            if line[-1] == "\n":
                lat = float(line[:-1].split()[0][:-1])
                lon = float(line[:-1].split()[1])
            else:
                lat = float(line.split()[0][:-1])
                lon = float(line.split()[1])
            geo_list.append({"latitude": lat,"longitude": lon,"accuracy": 10})

    #print(geo_list)

    driver = webdriver.Chrome()
    driver.get('https://www.retkikartta.fi')

    #simuloidaan arduino dataa
    i = 0
    rising = True
    jono.put(i)
    while True:
        varygeo_lat = geo_list[i]["latitude"]+(random.randint(0,1000)/10000)
        varygeo_lon = geo_list[i]["longitude"]+(random.randint(0,2000)/10000)
        #print(varygeo_lat,varygeo_lon)

        lat_multi.value = varygeo_lat
        lon_multi.value = varygeo_lon

        #print(geo_list[i])
        time.sleep(0.1)

        rand_geo_dict = {"latitude": varygeo_lat,"longitude": varygeo_lon,"accuracy": 10}
        driver.execute_cdp_cmd("Page.setGeolocationOverride", rand_geo_dict)


        #if rising and i == len(geo_list) - 1:
            #rising = False
        #if not rising and i == 0:
            #rising = True
        #if rising:
            #i += 1
        #else:
            #i -= 1

if __name__ == "__main__":
    multiprocessing.freeze_support()

    jono = multiprocessing.Queue()#etta venataan gps signaalia

    lat_multi = multiprocessing.Value('d', 0.0)
    lon_multi = multiprocessing.Value('d', 0.0)

    p2 = multiprocessing.Process(target=arduino_simulaattori, args=(lat_multi, lon_multi, jono,))
    p2.start()


    main(lat_multi, lon_multi, jono)
    p2.terminate()

    #p2 terminoidaan eniweis mainin lopussa... gottabee there?
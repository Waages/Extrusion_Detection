% COM-Port und Baudrate anpassen
s = serialport("COM6", 9600);  

configureTerminator(s, "LF");  % ESP32 sendet wahrscheinlich \n
flush(s);  % Puffer leeren

Samplesize = 3000;

Wert = zeros(Samplesize,1);

dist = input("Messabstand: ");

tic;
i = 1;
while i <= Samplesize
    if s.NumBytesAvailable > 0
        dataStr = readline(s);
        val = str2double(dataStr);
        if ~isnan(val)
            Wert(i)=val;
            if mod(i,Samplesize/10) == 0
                disp (string(i/Samplesize * 100) + "%");
            end
            i=i+1;
            
            %app = smooth(val,100);
            %addpoints(h, toc, val);
            %addpoints(h, toc, val);
            %drawnow limitrate;
        end
    end
end
Zeit = toc;

clear s;  % Verbindung schließen

%Histogramm erstellen
figure; %('Visible', 'off');
histogram(Wert,min(Wert):1:max(Wert));
xlabel("Wert");
ylabel("Häufigkeit");
%title("Histogramm der Messdaten");
grid on;

% Ergebnisse berechnen
fMess = Samplesize / Zeit;
mittl = mean(Wert);
stabw = std(Wert);

disp("fMess = " + string(fMess) + " Hz");
disp("Sampl = " + string(Samplesize));
disp("Mittl = " + mittl);
disp("Stabw = " + stabw);

%Kommentar
comment = input("Kommentar: ","s");

% --- Ordner erstellen, falls nicht vorhanden ---
outputFolder = "Messdaten";
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% --- CSV-Datei vorbereiten ---
csvFile = fullfile(outputFolder, "Messdaten.csv");

% Index bestimmen
if isfile(csvFile)
    existingData = readtable(csvFile);
    index = height(existingData) + 1;
else
    index = 1;
end

% Neue Zeile vorbereiten
newRow = {index, fMess, Samplesize, mittl, stabw, dist,comment};

% In CSV-Datei schreiben (anhängen)
fid = fopen(csvFile, 'a');
if index == 1
    fprintf(fid, "Index,fMess [Hz],Samples,Mittelwert,Standardabw,Entfernung,Kommentar\n");
end
fprintf(fid, "%d,%.2f,%d,%.4f,%.4f,%d,%s\n", ...
        newRow{1}, newRow{2}, newRow{3}, newRow{4}, newRow{5}, newRow{6}, newRow{7});
fclose(fid);


% --- Histogramm speichern ---
histName = fullfile(outputFolder, sprintf("histogram_%03d.png", index));
saveas(gcf, histName);
close(gcf);
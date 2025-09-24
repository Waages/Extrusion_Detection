datei = 'TMESS1.csv';
nummer = sscanf(datei, 'TMESS%d.csv');

% Tabelle auslesen, dabei die ersten 31 Zeilen ignorieren
opts = detectImportOptions(datei, 'NumHeaderLines', 31);
dataTab = readtable(datei, opts);


%%%% BURNOUT und letzten Spalten ignorieren %%%%%
% Erste Datenzeile extrahieren
firstRow = dataTab(1, :);

% Initialisiere logischen Vektor für "BURNOUT"-Spalten
isBurnout = false(1, width(dataTab));

% Iteriere über alle Spalten
for i = 1:width(dataTab)
    val = firstRow{1, i};
    % Vergleiche als String (robust gegenüber verschiedenen Typen)
    isBurnout(i) = strcmpi(strtrim(string(val)), 'BURNOUT');
end

% Letzte drei Spalten entfernen
nCols = width(dataTab);
colsToRemove = isBurnout | (1:nCols > nCols - 3) | (1:nCols == 1);

% Relevante Spalten behalten
dataTab(:, colsToRemove) = [];



%%%% Relativen Zeitstempel setzen %%%%%
% Zeitstempel als datetime parsen
timeStrings = string(dataTab{:, 1});  % Annahme: Zeit ist jetzt in Spalte 1
timeBase = datetime(timeStrings, 'InputFormat', 'yyyy/MM/dd HH:mm:ss');

% Millisekunden hinzufügen
ms = dataTab{:, 2};  % Annahme: Millisekunden sind jetzt in Spalte 2
timeWithMs = timeBase + milliseconds(ms);

% Relativer Zeitvektor in Sekunden zur ersten Zeile
t_rel = seconds(timeWithMs - timeWithMs(1));

% Ursprüngliche Zeitspalten entfernen
dataTab(:, 1:2) = [];

% Relative Zeit als erste Spalte einfügen
dataTab = addvars(dataTab, t_rel, 'Before', 1, 'NewVariableNames', 'Zeit');

% Messfrequenz auslesen aus zweitem Messwert
fMess = 1/table2array(dataTab(2,1));

% Initialisiere leeres Zell-Array mit gleicher Größe wie data
dataStr = strings(size(dataTab));

% Alle Spalten einzeln in Strings umwandeln
for i = 1:width(dataTab)
    col = dataTab{:, i};  % hole Spalte unabhängig vom Typ
    dataStr(:, i) = string(col);  % konvertiere alles in Strings
end

% Daten aufräumen
dataStr = replace(dataStr, "+ ", "");
dataStr = replace(dataStr, ",", ".");
dataStr = strtrim(dataStr);

% In double konvertieren
data = str2double(dataStr);
%%%% VORBEREITUNG FERTIG %%%%

plot(data(:,1),data(:,2:end));


%%% Startzeit festlegen
StartZeileGef = false;
StartZeile = 1;
Start = 1;
while(StartZeileGef == false)
    Start = input("Startzeit:");
    disp(Start);
    if (isempty(string(Start)))
        StartZeileGef = true;
    elseif Start < 1 || Start > size(data, 1)
        disp("ungültige Zeilennummer");
    else
        StartZeile = Start * fMess;
        plot(data(StartZeile:end,1),data(StartZeile:end,2:end));
    end
end

disp("Startzeit festgelegt");

%%% Endzeit festlegen
EndZeileGef = false;
EndZeile = length(data);
Ende = data(end,1);
while(EndZeileGef == false)
    Ende = input("Endzeit:");
    disp(Ende);
    if (isempty(string(Ende)))
        EndZeileGef = true;
    elseif Ende * fMess < StartZeile || Ende > size(data, 1)
        disp("ungültige Zeilennummer");
    else
        EndZeile = Ende * fMess;
        plot(data(StartZeile:EndZeile,1),data(StartZeile:EndZeile,2:end));
    end
end

FeinTuning = true;
while (FeinTuning)
    FT = input ("Feintuning? (Y/n)",'s');
    if FT == "n"
        FeinTuning = false;
    elseif FT == "Y"
        StartZeileGef = false;
        Start = StartZeile / fMess;
        while(StartZeileGef == false)
            Start = input("Startzeit (aktuell " + string(Start) + "):");
            disp(Start);
            if (isempty(string(Start)))
                StartZeileGef = true;
            elseif Start < 1 || Start > size(data, 1)
                disp("ungültige Zeilennummer");
            else
                StartZeile = Start * fMess;
                plot(data(StartZeile:EndZeile,1),data(StartZeile:EndZeile,2:end));
            end
        end
        EndZeileGef = false;
        Ende = EndZeile / fMess;
        while(EndZeileGef == false)
            Ende = input("Endzeit (aktuell " + string(Ende) + "):");
            disp(Ende);
            if (isempty(string(Ende)))
                EndZeileGef = true;
            elseif Ende * fMess < StartZeile || Ende > size(data, 1)
                disp("ungültige Zeilennummer");
            else
                EndZeile = Ende * fMess;
                plot(data(StartZeile:EndZeile,1),data(StartZeile:EndZeile,2:end));
            end
        end
    end
end

% Daten zum Auswerten ablegen
datanew = data(StartZeile:EndZeile,:);

disp("Fertig");

writematrix(datanew,"Data.csv");





%histogram(datanew(:,2:end));

figure;
hold on;             

edges = round(min(datanew(:,2:end))):1:round(max(datanew(:,2:end)));

for i = 2:size(datanew, 2)
    histogram(datanew(:, i),edges);
end






% Messpunktkoordinaten festlegen
if (nummer == 1 || nummer == 2)
    x = [30, 70, 110, 70, 110]; 
    y = [10, 10, 10, 50, 50];
elseif (nummer == 3 || nummer == 4)
    x = [70, 110, 70, 110, 70, 110]; 
    y = [50, 50, 90, 90, 130, 130];
elseif (nummer == 5)
    x = [30, 70, 110, 70, 110, 70, 110, 70, 110, 70, 110]; 
    y = [10, 10, 10, 50, 50, 50, 50, 90, 90, 130, 130];

else
    disp("Koordinaten nicht vergeben");
end


function T = threshold_max_consecutive(data, n)
    % data: Matrix, jede Spalte wird separat betrachtet
    % n: Mindestanzahl aufeinanderfolgender Werte >= Schwellenwert

    [~, numCols] = size(data);
    T = NaN(1, numCols);  % Ergebnis-Vektor

    for col = 1:numCols
        columnData = data(:, col);
        uniqueVals = unique(columnData, 'sorted');  % aufsteigend
        uniqueVals = flip(uniqueVals);  % absteigend sortieren

        % Gehe Schwellenwerte von hoch nach niedrig durch
        for thresh = uniqueVals'
            binVec = columnData >= thresh;  % 1 wenn >= Schwelle, sonst 0
            consecCount = max_consecutive_ones(binVec);
            if consecCount >= n
                T(col) = thresh;
                break;  % höchste gültige Schwelle gefunden
            end
        end
    end
end

function maxCount = max_consecutive_ones(vec)
    % Hilfsfunktion: Zählt längste Serie von 1en
    d = diff([0; vec(:); 0]);  % Anfügen von 0en am Rand
    runStarts = find(d == 1);
    runEnds = find(d == -1);
    runLengths = runEnds - runStarts;
    if isempty(runLengths)
        maxCount = 0;
    else
        maxCount = max(runLengths);
    end
end


function T = max_multiples(data, n)
    % data: Matrix, jede Spalte wird separat betrachtet
    % n: Mindestanzahl Werte >= Schwellenwert

    [~, numCols] = size(data);
    T = NaN(1, numCols);  % Ergebnis-Vektor

    for col = 1:numCols
        columnData = data(:, col);
        uniqueVals = unique(columnData, 'sorted');  % aufsteigend
        uniqueVals = flip(uniqueVals);  % absteigend sortieren

        T(col) = uniqueVals(n);
    end
end



%%% wie viele max hintereinander
T = threshold_max_consecutive(datanew(:,2:end),2);





% --- Bereich definieren (ab 0,0 starten) ---
xq = linspace(0, max(x), 100);
yq = linspace(0, max(y), 100);
[xq, yq] = meshgrid(xq, yq);

% --- Interpolation ---
Tq = griddata(x, y, T, xq, yq, 'cubic');

% --- Plot ---
figure;
contourf(xq, yq, Tq, 20, 'LineColor', 'none');
colormap('jet');
colorbar;
xlabel('X [mm]');
ylabel('Y [mm]');
title('Temperaturkarte');

% --- Achsen korrekt setzen ---
axis([0 max(x) 0 max(y)]); % Bereich auf (0,0) bis (max(x), max(y))
axis equal tight;

% --- Figurgröße exakt einstellen ---
set(gcf, 'Units', 'centimeters');
x_width_cm = max(x) / 10; % mm -> cm
y_width_cm = max(y) / 10;
set(gcf, 'Position', [0 0 x_width_cm y_width_cm]);

% --- Achsen auf die gesamte Figur ausdehnen ---
set(gca, 'Units', 'centimeters');
set(gca, 'Position', [0 0 x_width_cm y_width_cm]); 

% Achsenticks und Schriftgröße anpassen (klein halten, damit es nicht stört)
set(gca, 'FontSize', 6);

% --- Exportieren ohne Rand ---
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 x_width_cm y_width_cm]);
set(gcf, 'PaperSize', [x_width_cm y_width_cm]);

% --- Export als Vektorformat (PDF) ---
% print('Temperaturkarte_export', '-dpdf', '-r300');
% oder alternativ:
saveas(gcf, 'Temperaturkarte_export.svg');
saveas(gcf, 'Temperaturkarte_export.png');


disp("Max Total")
disp(max(datanew(:,2:end)));

disp("Max 5");
disp(max(max_multiples(datanew(:,2:end),round(5*fMess))));

disp("Max 10");
disp(max(max_multiples(datanew(:,2:end),round(10*fMess))));

disp("Max 20");
disp(max(max_multiples(datanew(:,2:end),round(20*fMess))));

disp("Max Con 5");
disp(max(threshold_max_consecutive(datanew(:,2:end),round(5*fMess))));

disp("Max Con 10");
disp(max(threshold_max_consecutive(datanew(:,2:end),round(10*fMess))));

disp("Max Con 20");
disp(max(threshold_max_consecutive(datanew(:,2:end),round(20*fMess))));



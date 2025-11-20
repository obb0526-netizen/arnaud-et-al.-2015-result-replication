% Matlab script to load all subject
% animal/non-animal picture from session 1
%
% Arnaud Delorme - Feb 2007
% ========================================
clear
eeglab; close
tempFolder = '../eeglab_data';
mkdir(tempFolder);

%% participant information for participants.tsv file
% -------------------------------------------------
pInfo = { 'gender' };
gender = { 'cba' 'F'; % already the correct order
           'clm' 'F';
           'ega' 'F';
           'fsa' 'M';
           'gro' 'M';
           'hth' 'M';
           'lmi' 'F';
           'mba' 'F';
           'mma' 'M';
           'mta' 'F';
           'pla' 'M';
           'sce' 'F';
           'sph' 'M';
           'wpa' 'M' };
           
alls = dir('../raw');
alls = { alls.name };
indRm = cellfun(@(x)(x(1) == '.'), alls);
alls(indRm) = [];
alls = setdiff(alls, 'README.txt');
alls = setdiff(alls, 'channel_location_file.loc');

usertags = {};
usertags{121} = 'Event/Category/Experimental stimulus, Event/Label/animal_target, Event/Description/Target animal image in go no-go task, Attribute/Location/Screen/Center, Attribute/Visual/Background, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Target, Sensory presentation/Visual/Rendering type/Screen';
usertags{122} = 'Event/Category/Experimental stimulus, Event/Label/easy_target, Event/Description/Single easy animal target image in go no-go task, Attribute/Location/Screen/Center, Attribute/Visual/Background, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Target, Sensory presentation/Visual/Rendering type/Screen';
usertags{123} = 'Event/Category/Experimental stimulus, Event/Label/difficult_target, Event/Description/Single difficult animal target image in go no-go task, Attribute/Location/Screen/Center, Attribute/Visual/Background, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Target, Sensory presentation/Visual/Rendering type/Screen';
usertags{124} = 'Event/Category/Experimental stimulus, Event/Label/nonanimal_target, Event/Description/Single non-animal target image in go no-go task, Attribute/Location/Screen/Center, Attribute/Visual/Background, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Target, Sensory presentation/Visual/Rendering type/Screen';
usertags{126} = 'Event/Category/Experimental stimulus, Event/Label/Non-animalDistrator, Event/Description/Distractor non-animal image in go-nogo categorization task, Attribute/Location/Screen/Center, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Expected/Distractor, Sensory presentation/Visual/Rendering type/Screen';
usertags{127} = 'Event/Category/Experimental stimulus, Event/Label/Non-animalDistrator, Event/Description/Distractor non-animal image in go-nogo easy animal image detection task, Attribute/Location/Screen/Center, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Expected/Distractor, Sensory presentation/Visual/Rendering type/Screen';
usertags{128} = 'Event/Category/Experimental stimulus, Event/Label/Non-animalDistrator, Event/Description/Distractor non-animal image in go-nogo difficult animal image detection task, Attribute/Location/Screen/Center, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Expected/Distractor, Sensory presentation/Visual/Rendering type/Screen';
usertags{129} = 'Event/Category/Experimental stimulus, Event/Label/Non-animalDistrator, Event/Description/Distractor non-animal image in go-nogo non-animal image detection task, Attribute/Location/Screen/Center, Item/Natural scene, Paradigm/Go-no-go task, Participant/Effect/Cognitive/Expected/Distractor, Sensory presentation/Visual/Rendering type/Screen';
usertagsresp  = 'Event/Category/Participant response, Event/Label/ReactionTime, Event/Description/Subject reaction time, Action/Button release';

count = 1;
data  = [];
mkdir(fullfile(tempFolder, 'sourcedata'));
copyfile('../raw/README.txt', fullfile(tempFolder, 'sourcedata'), 'f');
copyfile('../raw/channel_location_file.loc', fullfile(tempFolder, 'sourcedata'), 'f');
for iSubj = 1:length(alls)
    
    if ~ismember( alls{iSubj}, gender(:,1))
        error('Unknown subject');
    else
        [~,loc] = ismember( alls{iSubj}, gender(:,1));
        pInfo = [ pInfo; gender(loc,2) ];
    end
    mkdir(fullfile(tempFolder, 'sourcedata', alls{iSubj}));
    data(iSubj).subject  = sprintf('sub-%3.3d', iSubj);
    data(iSubj).chanlocs = '../raw/channel_location_file.loc';
            
    count2 = 1;
    for iSess = 1:2
        for iRun = 1:fastif(iSess == 1, 13, 12)
            
            % raw data files
            baseNameIn = sprintf('../raw/%s/%s1ff%2.2d',alls{iSubj}, alls{iSubj}, iRun);
            if ~exist( [baseNameIn '.cnt' ] )
                error(' ');
            end
            
            % copy data to BIDS
            baseNameEEGLAB = sprintf('%s/sub-%3.3d_sess%2.2d_run%2.2d.set', tempFolder, iSubj, iSess, iRun);
            data(iSubj).file(count2).file     = baseNameEEGLAB;
            data(iSubj).file(count2).session  = iSess;
            data(iSubj).file(count2).run      = iRun;
            baseNameOut = fullfile(tempFolder, 'sourcedata', alls{iSubj}, sprintf('sub-%3.3d_sess%2.2d_run%2.2d', iSubj, iSess, iRun));
            copyfile( [baseNameIn '.cnt' ], [baseNameOut '.cnt' ], 'f');
            copyfile( [baseNameIn '.dat' ], [baseNameOut '.dat' ], 'f');
            copyfile( [baseNameIn '.exp' ], [baseNameOut '.exp' ], 'f');
            
            % import data
            EEG  = pop_loadcnt([baseNameIn '.cnt'], 'dataformat', 'int16');
            res  = loadtxt([baseNameIn '.dat' ], 'skipline', 20);
            res2 = loadtxt([baseNameIn '.exp' ], 'skipline', 8, 'delim', 9);
            if res{1,1} == 1, discrepency = 0; else discrepency = -1; end
            if size(res,1) ~= size(res2,1), error('Difference between dat and exp files'); end
            
            % remove boundaries, add reaction time and event information
            for index = 1:length(EEG.event)
                EEG.event(index).type = num2str(EEG.event(index).type);
            end
            indbnd = [];
            offset = 0;
            data(count).file(count2).tasktype = [];
            for index = 1:length(EEG.event)
                if strcmpi(EEG.event(index).type, 'boundary')
                    indbnd = [ indbnd index ];
                    offset = offset-1;
                else
                    if index+offset > size(res,1)
                        disp('MISSING RECORD');
                    else
                        if ~(str2num(EEG.event(index).type)+discrepency == res{index+offset,1})
                            offset = offset+1;
                            disp('SKIPPING ONE LINE');
                        end
                        if ~(str2num(EEG.event(index).type)+discrepency == res{index+offset,1})
                            error('Wrong index');
                        end
                        code = ceil(res{index+offset,3}/100);
                        switch code
                            case 121, EEG.event(index).value = 'animal_target';        if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 1; elseif data(count).file(count2).tasktype ~= 1, error(' '); end
                            case 122, EEG.event(index).value = 'easy_target';          if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 2; elseif data(count).file(count2).tasktype ~= 2, error(' '); end
                            case 123, EEG.event(index).value = 'difficult_target';     if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 3; elseif data(count).file(count2).tasktype ~= 3, error(' '); end
                            case 124, EEG.event(index).value = 'nonanimal_target';     if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 4; elseif data(count).file(count2).tasktype ~= 4, error(' '); end
                            case 126, EEG.event(index).value = 'animal_distractor';    if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 1; elseif data(count).file(count2).tasktype ~= 1, error(' '); end
                            case 127, EEG.event(index).value = 'easy_distractor';      if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 2; elseif data(count).file(count2).tasktype ~= 2, error(' '); end
                            case 128, EEG.event(index).value = 'difficult_distractor'; if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 3; elseif data(count).file(count2).tasktype ~= 3, error(' '); end
                            case 129, EEG.event(index).value = 'nonanimal_distractor'; if isempty(data(count).file(count2).tasktype), data(count).file(count2).tasktype = 4; elseif data(count).file(count2).tasktype ~= 4, error(' '); end
                            otherwise
                                error('Unknown code');
                        end
                        EEG.event(index).usertag       = usertags{code}; 
                        EEG.event(index).trial_num     = EEG.event(index).type;
                        EEG.event(index).stim_file     = [ num2str(res2{index+offset,3}) '.jpg' ];
                        EEG.event(index).response_time = res2{index+offset,7};
                        EEG.event(index).trial_type    = fastif(res2{index+offset,2}, 'go', 'nogo');
                        EEG.event(index).correct       = res{index+offset,end-1};
                        EEG.event(index).trial_type    = 'stimulus';
                        if res{index+offset,end} ~= 1000 % reaction time
                            EEG.event(end+1).type     = 'response';
                            EEG.event(end).trial_type = 'response';
                            EEG.event(end).latency = EEG.event(index).latency+res{index+offset,end}/1000*EEG.srate;
                            EEG.event(end).correct    = res{index+offset,end-1};
                            EEG.event(end).usertag    = usertagsresp;
                            if res{index+offset,end-1}
                                EEG.event(end).value = 'correct';
                            else 
                                EEG.event(end).value = 'incorrect';
                            end
                        else
                            EEG.event(index).response_time = 'n/a';
                        end
                    end
                end
            end
            EEG.event(indbnd) = [];
            [~,indEvent] = sort([ EEG.event.latency ]);
            EEG.event = EEG.event(indEvent); % events resorted by latencies
            pop_saveset(EEG, 'filename', baseNameEEGLAB);

            % set task type
            switch data(count).file(count2).tasktype
                case 1
                    data(iSubj).file(count2).task = 'Go no-go animal categorization task';
                    data(iSubj).file(count2).instructions = 'Approximate instruction: release your finger as fast as possible from the optical button whenever you see an animal.';
                case 2
                    data(iSubj).file(count2).task = 'Go no-go single easy image detection task among varied distractors';
                    data(iSubj).file(count2).instructions = 'Approximate instruction: release your finger as fast as possible from the optical button whenever you see the target image.';
                case 3
                    data(iSubj).file(count2).task = 'Go no-go single difficult image detection task among varied distractors';
                    data(iSubj).file(count2).instructions = 'Approximate instruction: release your finger as fast as possible from the optical button whenever you see the target image.';
                case 4
                    data(iSubj).file(count2).task = 'Go no-go single difficult image detection task among varied distractors';
                    data(iSubj).file(count2).instructions = 'Approximate instruction: release your finger as fast as possible from the optical button whenever you see the target image.';
                otherwise
                    error('Unknown type');
            end
            count2 = count2+1;
            
        end
        
    end
    
end
save -mat precomputed_bids.mat data pInfo tempFolder;

%% Code Files used to preprocess and import to BIDS
% -----------------------------------------------------|
codefiles = {'/data/data/STUDIES/animal_na_study/matlab/bids_export_animal.m' };

% general information for dataset_description.json file
% -----------------------------------------------------
generalInfo.Name = 'Go-nogo categorization and detection task';
generalInfo.ReferencesAndLinks = { "https://www.ncbi.nlm.nih.gov/pubmed/11244543" "https://www.ncbi.nlm.nih.gov/pubmed/15019707" "https://papers.cnl.salk.edu/PDFs/From%20Single-Trial%20EEG%20to%20Brain%20Area%20Dynamics%202002-3661.pdf"};
generalInfo.BIDSVersion = 'v1.2.1';
generalInfo.License = 'CC0';
generalInfo.Authors = {'Arnaud Delorme' 'Michele Fabre-Thorpe' };

% participant column description for participants.json file
% ---------------------------------------------------------
pInfoDesc.participant_id.LongName    = 'Participant identifier';
pInfoDesc.participant_id.Description = 'Unique participant identifier';

pInfoDesc.gender.Description = 'Sex of the participant';
pInfoDesc.gender.Levels.M    = 'male';
pInfoDesc.gender.Levels.F    = 'female';

% 
% pInfoDesc.age.Description = 'age of the participant';
% pInfoDesc.age.Units       = 'years';

% event column description for xxx-events.json file (only one such file)
% ----------------------------------------------------------------------
eInfo = {'onset'         'latency';
         'value'         'type';
         'trial_type'    'trial_type'; 
         'stim_type'     'stim_type';
         'response_time' 'response_time';
         'correct'       'correct';
         'trial_num'     'trial_num';
         'HED'           'usertag' }; % ADD HED HERE

eInfoDesc.onset.Description = 'Event onset';
eInfoDesc.onset.Units = 'second';
                                
eInfoDesc.value.Description = 'Value of event';
eInfoDesc.value.Levels.animal_target     = 'An image containing an animal that is a target in a go-nogo animal categorization task';
eInfoDesc.value.Levels.easy_target       = 'An image containing an animal (easy to detect) that is a target in a single target image detection task';
eInfoDesc.value.Levels.difficult_target  = 'An image containing an animal (hard to detect) that is a target in a single target image detection task';
eInfoDesc.value.Levels.nonanimal_target  = 'An image not containing an animal that is a target in a single target image detection task';

eInfoDesc.value.Levels.animal_distractor    = 'An image not containing an animal that is a distractor in a go-nogo animal categorization task';
eInfoDesc.value.Levels.easy_distractor      = 'An image not containing an animal that is a distractor in a single target easy animal image detection task';
eInfoDesc.value.Levels.difficult_distractor = 'An image not containing an animal that is a distractor in a single target difficult animal image detection task';
eInfoDesc.value.Levels.nonanimal_distractor = 'An image not containing an animal that is a distractor in a single target non-animal image detection task';
eInfoDesc.value.Levels.correct              = 'Correct response';
eInfoDesc.value.Levels.incorrect            = 'Incorrect response';

eInfoDesc.trial_type.Description = 'Type of event';
eInfoDesc.trial_type.Levels.go   = 'Go-type trial in go-no task';
eInfoDesc.trial_type.Levels.nogo = 'No-go type trial in go-no task';
eInfoDesc.trial_type.Levels.responses = 'Behavioral response';

eInfoDesc.stim_type.Description = 'Stimulus image presented';

eInfoDesc.response_time.Description = 'Reaction time';
eInfoDesc.response_time.Units = 'millisecond';

eInfoDesc.correct.Description = 'Correct or incorrect responses';
eInfoDesc.correct.Levels.x0  = 'Incorect response';
eInfoDesc.correct.Levels.x1  = 'Correct response';

% HED Tags
eInfoDesc.HED.Description = 'Hierarchical Event Descricptor';

% Content for README file
% -----------------------
README = sprintf( [ 'Participants seated in a dimly lit room at 110 cm from a computer screen piloted from a PC computer. Two tasks alternated: a categorization task and a recognition task. In both tasks, target images and non-target images were equally likely presented. Participants were tested in two recording phases. The first day was composed of 13 series, the second day of 12 series, with 100 images per series (see details of the series below). To start a series, subjects had to press a touch-sensitive button. A small fixation point (smaller than 0.1 degree of visual angle) was drawn in the middle of a black screen. Then, an 8 bit color vertical photograph (256 pixels wide by 384 pixels high which roughly correspond to 4.5 degree of visual angle in width and 6.5 degree in height) was flashed for 20 ms (2 frames of a 100 Hz SVGA screen) using a programmable graphic board (VSG 2.1, Cambridge Research Systems). This short presentation time avoid that subjects use exploratory eye movement to respond. Participants gave their responses following a go/nogo paradigm. For each target, they had to lift their finger from the button as quickly and accurately as possible (releasing the button restored a focused light beam between an optic fiber led and its receiver; the response latency of this apparatus was under 1 ms). Participants were given 1000 ms to respond, after what any response was considered as a nogo response. The stimulus onset asynchrony (SOA) was 2000 ms plus or minus a random delay of 200 ms. For each distractor, participants had to keep pressing the button during at least 1000 ms (nogo response).' 10 10 ...
    'More specifically, in the animal categorization task, participants had to respond whenever there was an animal in the picture. In the recognition task, the session started with a learning phase. A probe image was flashed 15 times during 20 ms intermixed with two presentations of 1000 ms after the fifth and the tenth flashes, allowing an ocular exploration of the image; with an inter-stimulus of 1000 ms. Participants were instructed to carefully examine and learn the probe image in order to recognize it in the following series. The test phase started immediately after the learning phase. The probe image constituted the unique target of the series. Both tasks were organized in series of 100 images; 50 targets images were mixed with 50 non-targets in the animal categorization task; 50 copies of an unique photographs were mixed at random with 50 non-targets in the recognition task.' ]);

% Content for CHANGES file
% ------------------------
CHANGES = sprintf([ 'Revision history for categorization/detection dataset\n\n' ...
    'version 1.0 - 26 Feb 2020\n' ...
    ' - Initial release\n' ]);

% Task information for xxxx-eeg.json file
% ---------------------------------------
tInfo.InstitutionAddress = 'Pavillon Baudot CHU Purpan, BP 25202, 31052 Toulouse Cedex';
tInfo.InstitutionName = 'Paul Sabatier Universite';
tInfo.InstitutionalDepartmentName = 'Centre de Recherche Cerveau et Cognition';
tInfo.PowerLineFrequency = 50;
tInfo.ManufacturersModelName = 'Neuroscan Synamps 1 (model 5083)';

% call to the export function
% ---------------------------
targetFolder =  '../BIDS';
bids_export(data, 'targetdir', targetFolder, 'taskName', 'gonogo', 'gInfo', generalInfo, 'pInfo', pInfo, 'pInfoDesc', pInfoDesc, 'eInfoDesc', eInfoDesc, 'README', README, 'CHANGES', CHANGES, 'codefiles', codefiles, 'tInfo', tInfo, 'copydata', 1);

% copy stimuli folder
% -------------------
copyfile('../stimuli', fullfile(targetFolder, 'stimuli'), 'f');
copyfile(fullfile(tempFolder, 'sourcedata'), fullfile(targetFolder, 'sourcedata'), 'f');

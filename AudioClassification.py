import argparse
import csv
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torchaudio.transforms as T
from openai import OpenAI
from transformers import AutoProcessor, ClapModel
from tqdm import tqdm
import numpy as np

# Suppress HTTP request logging from OpenAI and urllib3
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Also suppress through environment variable
os.environ["OPENAI_LOG"] = "warning"
os.environ["HTTPX_LOG_LEVEL"] = "WARNING"

# Constants and word lists
TYPE_WORDS = ["stab", "riser", "pluck"]
NOISE_WORDS = ["noise", "signal"]
INSTRUMENT_WORDS = ["Accordion","Acoustic Guitar","Alto Saxophone","Bagpipes","Banjo","Bass Guitar","Bassoon","Bb Clarinet","Cello","Conga","Contrabassoon","Cornet","Cowbell","Didgeridoo","Djembe","Double Bass","Dulcimer","Electric Guitar","English Horn","Fiddle","Flugelhorn","Flute","French Horn","Gong","Grand Piano","Guitar","Harmonica","Harp","Harpsichord","Heavydrum","High-hat","Kalimba","Koto","Lute","Mandolin","Maracas","Marimba","Melodica","Mellotron","Moog Synthesizer","Oboe","Organ","Pan Flute","Pedal Steel Guitar","Piccolo","Piano","Piccolo Trumpet","Recorder","Rhodes Piano","Sitar","Snare Drum","Steel Drum","Steel Guitar","Steelpan","Synthesizer","Tabla","Tambourine","Tenor Saxophone","Theremin","Timpani","Timbales","Tom-Tom","Trombone","Trumpet","Tuba","Ukulele","Viola","Violin","Vibraphone","Washboard","Whistle","Xylophone","Zither","Aeolian Harp","Balalaika","Bass Clarinet","Bass Drum","Bass Recorder","Bass Trombone","Bass Trumpet","Bell Lyre","Bodhrán","Bouzouki","Cajón","Castanets","Celesta","Chimes","Clarinet","Clavichord","Contrabass","Cuíca","Dizi","Dulcian","Ektara","English Bagpipes","Fife","Flexatone","Gamelan","Glass Harmonica","Glockenspiel","Gong Ageng","Gongs","Guiro","Hang Drum","Hardanger Fiddle","Harp Guitar","Hurdy-Gurdy","Jaw Harp","Kalangu","Kantele","Kazoo","Kettle Drum","Khlui","Kobza","Komungo","Lirone","Mandola","Mbira","Mellophone","Melodion","Mizmar","Moog","Morin Khuur","Musical Saw","Ney","Nyckelharpa","Ocarina","Octoban","Ondes Martenot","Orchestral Bells","Oud","Panpipes","Paraguayan Harp","Percussion","Piano Accordion","Piccolo Clarinet","Pipe Organ","Pipa","Poitou Bagpipes","Quena","Quijada","Ravanahatha","Rebab","Resonator Guitar","Riq","Robab","Sansa","Sarangi","Sarrusophone","Saxophone","Sheng","Shofar","Shakuhachi","Shehnai","Siku","Slit Drum","Snare","Sousaphone","Spoons","Steel Tongue Drum","Suona","Tabla Tarang","Taiko","Tambura","Tanggu","Tar","Tarogato","Timpano","Tonette","Tumbi","Turruca","Valiha","Veena","Vielle","Viol","Violone","Xaphoon","Yangqin","Zhaleika","Zurna","Zymbal","Agogo","Aluphone","Anklung","Angklung","Bass Flute","Bass Violin","Bendir","Berimbau","Bianzhong","Bombard","Bouzouki Mandolin","Brazilian Cuica","Bugle","Cajon","Caxixi","Chinese Flute","Claves","Cuatro","Daf","Darbuka","Daxophone","Dhol","Dholak","Digeridoo","Dizi Flute","Ektar","Flügelhorn","Frula","Ganassi Recorder","Garklein Recorder","Guitarrón","Guqin","Handpan","Hardanger Fiddle","Hapi Drum","Hichiriki","Hulusi","Irish Flute","Irish Harp","Jembe","Jew's Harp","Kacapi","Kamancheh","Kanjira","Kaval","Kayagum","Kendang","Khen","Kora","Krar","Kundi","Lusheng","Madal","Mrdanga","Moyle","Mridangam","Ngoni","Niwa","Oboe d’amore","Octobans","Ocarina","Ondioline","Pan Flute","Parang","Piri","Pizzicato","Pump Organ","Qin","Reco-reco","Renaissance Flute","Riqq","Riq","Sarod","Shakuhachi","Sheng","Shime-daiko","Siku","Sitar","Sousaphone","Steelpan","Suona","Surbahar","Taiko Drum","Tarogato","Tambour","Tangmuri","Tarhu","Theremin","Tumbi","Valiha","Vibra-slap","Waterphone","Whamola","Wood Block","Yangqin","Zither"]
TONE_QUALITY_WORDS = ["airy","alien","angular","annoying","antique","arid","articulate","assertive","astringent","atmospheric","attenuated","baffled","bassy","beautiful","beefy","biting","bizarre","bland","bleak","blurry","boomy","boxy","brash","bright","brilliant","brittle","broken","bubbly","burly","burred","bursty","buzzy","candid","celestial","centered","chewy","chirpy","choked","clean","clear","clinical","clipped","cloudy","cluttered","coarse","coherent","compact","compressed","congested","controlled","creamy","crisp","crunchy","crystalline","cultured","dark","dead","dense","detached","dirty","dissonant","distant","dolorous","dreamy","driven","dry","dubby","dull","edgy","elastic","elegant","ethereal","evocative","even","faint","fat","feeble","fibrous","filtered","fizzy","flat","fluffy","focused","forceful","fragile","frazzled","fresh","fretful","frothy","fuzzy","gentle","ghostly","glasslike","glassy","gleaming","glossy","grainy","grating","gravelly","greasy","grimy","gritty","growly","harsh","hazy","hearty","heavy","hectic","hefty","hesitant","hissing","hollow","honky","hot","husky","hushed","icy","immediate","imprecise","incisive","incoherent","industrial","intense","intimate","jangly","jittery","juicy","keen","lacy","laid-back","lame","languid","lean","leathery","light","limp","liquid","loamy","loud","low","loyal","lucid","lush","masked","meaty","metallic","mild","moist","mournful","muddy","muffled","murky","nasal","neutral","noisy","nonlinear","obscure","odd","open","organic","overdriven","overripe","pale","peaky","percussive","piercing","plain","plangent","plastic","plucky","pointed","popcorny","powdery","precise","prickly","processed","pronounced","punchy","pure","quacky","quiet","ragged","raucous","raw","razored","resonant","rich","ringing","ripe","rough","rounded","rumbling","runny","sandy","scathing","sculpted","searing","sharp","sheetlike","shrill","silky","silvery","sinister","slatey","slick","slimy","slinky","sloppy","slurred","smoky","smooth","snarling","snazzy","snappy","soggy","solid","sonorous","sparkling","sparse","spectral","spiky","spitty","splashy","splintered","squashed","squeaky","stable","stale","steely","sterile","sticky","stiff","strained","streamlined","stretchy","stringy","strong","subdued","sugary","swampy","sweeping","sweet","syrupy","taut","tense","thin","thick","throaty","tight","timbred","tinny","tiny","trembling","troubled","tubby","turgid","tweaked","twinkly","uneven","unfocused","unrefined","velvety","vibrant","vicious","vilny","vinegary","vivid","voiced","warm","washy","watery","weak","wet","whiny","whispy","whistling","wide","woody","woolly","worn","yelping","zappy","zingy"]
PITCH_LEVEL_WORDS = ["high","low","mid","very high","very low","ultra high","ultra low","extreme","deep","shrill","bright","dark","thin","thick","sharp","flat","narrow","wide","piercing","boomy","crisp","dull","clear","fuzzy","focused","unfocused","glassy","gritty","warm","cold","airy","nasal","throaty","open","closed","centered","detuned","tuned","resonant","non-resonant","tense","relaxed","subtle","harsh","round","pointed","full","hollow","light","heavy","compact","spacious","buzzy","smooth","rough","edgy","grainy","clean","dirty","modulated","unmodulated","even","uneven","steady","wavering","pitched","unpitched","gliding","static","fluctuating","ascending","descending","stable","unstable","climbing","fading","expanding","compressing","peaking","dipping","punchy","ghosted","accented","unaccented","rich","thin-bodied","throbbing","whining","whistling","rattling","twinkling","humming","barking","bubbling","chirping","clicking","clipping","clanging","crackling","crunchy","droning","fluttering","fizzing","glitchy","growling","hissing","honking","jittery","jumpy","lush","melodic","monotone","muted","oscillating","panning","pinging","plucked","pointillistic","popping","pulsing","pure","raspy","ringing","robotic","rumbling","scratchy","screechy","sizzling","snappy","sparkling","speaking","spectral","spiky","spitty","sputtery","staccato","strident","swelling","syncopated","ticking","tinny","trembling","tremolo","vibrating","vibrato","wailing","wavy","wet","wheezing","whirring","whumping","woody","wobbly","wow","zip","glissando","portamento","intervallic","tiny","giant","super low","super high","low-mid","high-mid","subsonic","subharmonic","ultrasonic","overtone","harmonic","inharmonic","fundamental","partial","formant","octave","sub-octave","top-end","bottom-end","low register","high register","head voice","chest voice","falsetto","whistle","modal","baritone","tenor","alto","soprano","bass","contralto","mezzo-soprano","glottal","breathy","constricted","projected","forward","backward","lateral","vertical","pinched","relaxed","strained","resolved","unresolved","leading","resting","tonal","atonal","keyed","unkeyed","dynamic","static","flexible","inflexible","controlled","chaotic","dissonant","consonant","elevated","grounded","sharped","flatted","transposed","unchanged","sloped","tapered","curved","gradual","abrupt","crested","valleyed","echoed","reflected","refracted","reverberant","ambient","dry","surged","bursting","subdued","intense","minor","major","neutral","modal mixture","diatonic","chromatic","microtonal","just intoned","well-tempered","equal-tempered","overblown","underblown"]
VOLUME_INTENSITY_WORDS = ["whisper", "murmur", "hush", "faint", "quiet", "soft", "subtle", "muted", "muffled", "low", "low-key", "subdued", "gentle", "mild", "airy", "delicate", "dim", "barely-audible", "reserved", "calm", "passive", "distant", "fading", "whispery", "shy", "slight", "feeble", "weak", "fainthearted", "breezy", "thin", "pale", "mellow", "fine", "flat", "hollow", "dry", "damped", "suppressed", "softened", "brushed", "light", "tiptoe", "reticent", "quiescent", "tranquil", "peaceful", "tame", "neutral", "inert", "docile", "restrained", "shadowy", "latent", "tentative", "evanescent", "modest", "bashful", "diffident", "elusive", "foggy", "washed-out", "dusky", "stifled", "hidden", "underplayed", "compressed", "subtlety", "decay", "silence", "murky", "fainted", "noiseless", "breathy", "demure", "indistinct", "vaporous", "whisper-soft", "nearly-inaudible", "attenuated", "airy-light", "smothered", "veiled", "understated", "unassuming", "indistinguishable", "forceless", "gasping", "filtered", "backgrounded", "ambient", "tranquilized", "tiptoed", "padded", "deadened", "smoothed", "lulling", "dimmed", "neutralized", "recessive", "breathless", "minimal", "downplayed", "swallowed", "softened-tone", "faraway", "faded", "echoic", "veiling", "quivering", "trembly", "airy-whisper", "ghostly", "repressed", "far-off", "indistinctive", "pale-sounding", "breath-thin", "whisperlike", "hush-toned", "subliminal", "distant-sounding", "calm-toned", "liminal", "dim-tone", "obscured", "semi-audible", "purring", "rumbling", "rounded", "warm", "controlled", "firm", "bold", "dynamic", "resonant", "expressive", "emphasized", "robust", "colored", "lively", "clear", "present", "noticeable", "articulate", "bright", "natural", "distinct", "forceful", "powered", "solid", "steady", "even", "tempered", "pressed", "broadcast", "unmuted", "free", "relaxed", "open", "confident", "singing", "rich", "clean", "accurate", "full", "fluent", "tonal", "sung", "normal", "harmonic", "soundful", "dry-loud", "talking", "echoing", "open-air", "audible", "well-spoken", "amplified", "lifted", "voiced", "aired", "expanded", "ringing", "carried", "commanding", "elevated", "soaring", "enhanced", "projected", "fronted", "far-reaching", "near-maximum", "energetic", "expansive", "overt", "pronounced", "intense", "sharp", "peaking", "blaring", "shrill", "brightened", "exposed", "unveiled", "vibrating", "thick", "broad", "strong", "assertive", "vibrant", "weighty", "throbbing", "pumping", "banging", "loud", "strong-toned", "hard", "raucous", "grating", "screaming", "brassy", "thundering", "heavy", "pounding", "roaring", "deafening", "explosive", "pealing", "thunderous", "maxed", "super-loud", "overwhelming", "shrieking", "howling", "tumultuous", "clashing", "crashing", "violent", "jarring", "bursting", "blinding", "brutal", "brutalist", "saturated", "max-out", "clipped", "overloaded", "distorted", "blown", "fierce", "harsh", "crushing", "wall-of-sound", "massive", "saturated-tone", "sharp-edged", "spine-rattling", "blistering", "squealing", "tearing", "raw", "force-maxed", "pain-inducing", "skull-shaking", "obliterating", "sonic-boom", "earthquaking", "seismic", "shattering"]
TIME_SHAPE_WORDS = ["flow", "rhythm", "loop", "spiral", "wave", "pulse", "beat", "interval", "sequence", "cycle", "arc", "duration", "moment", "stretch", "wrinkle", "fold", "curve", "bend", "echo", "delay", "sync", "drift", "stutter", "phase", "cadence", "period", "lag", "tempo", "acceleration", "deceleration", "instant", "recurrence", "recursion", "feedback", "jitter", "flux", "decay", "progression", "regression", "harmony", "dissonance", "layer", "pattern", "pulsewidth", "notch", "shift", "sweep", "modulation", "symmetry", "asymmetry", "lag", "warp", "timewarp", "timestream", "continuity", "break", "glitch", "ripple", "resonance", "vibration", "frequency", "oscillation", "frame", "tick", "tock", "timestamp", "chronology", "timeline", "era", "epoch", "age", "season", "dawn", "dusk", "zenith", "nadir", "twilight", "noon", "midnight", "now", "before", "after", "past", "present", "future", "eternity", "immediacy", "transience", "impermanence", "permanence", "intervallic", "fragmented", "seamless", "granular", "atomic", "elastic", "rigid", "quantized", "analog", "digital", "analogic", "temporal", "atemporal", "timeless", "phased", "nested", "concurrent", "consecutive", "spontaneous", "preordained", "stretched", "compressed", "layered", "overlaid", "interleaved", "juxtaposed", "sequenced", "infinite", "finite", "circular", "linear", "non-linear", "fractal", "branching", "converging", "diverging", "mirrored", "folded", "mapped", "encoded", "decoded", "aligned", "offset", "pivot", "anchor", "syncopated", "swinging", "swaying", "ticking", "thrumming", "reverberating", "humming", "whispering", "shifting", "sliding", "morphing", "shaping", "curling", "winding", "rising", "falling", "pulsing", "flickering", "fading", "lingering", "darting", "leaping", "pausing", "surging", "slowing", "speeding", "glitching", "tumbling", "coiling", "spiraling", "cascading", "orbiting", "rotating", "revolving", "compressing", "expanding", "dissolving", "coalescing", "unfolding", "enfolding", "emergent", "recurrent", "disjointed", "coherent", "chaotic", "ordered", "fluid", "static", "shifting", "transforming", "mapping", "reshaping", "pacing", "timing", "sequencing", "fracturing", "flowing", "blooming", "collapsing", "oscillating", "bouncing", "beating", "repeating", "iterating", "alternating", "syncing", "unfolding", "branching", "splitting", "ticking", "encoding", "recording", "tracking", "timestamping", "looping", "resolving", "fading", "echoing", "reverberating", "compressive", "expansive", "layered", "periodic", "dynamic", "relative", "absolute", "emergent", "nested", "recursive", "causal", "acausal", "premonitory", "reflective", "anticipatory", "anticipative", "transient", "cyclical", "prophetic", "retroactive", "spontaneous", "temporalized", "dimensioned", "sequenced", "woven", "intertwined", "shifting", "bounded", "measurable", "warped", "distorted", "shaped", "configured", "expressed", "abstracted", "notated", "felt", "sensed", "modeled", "perceived", "sculpted", "interpreted", "calculated", "expressed", "imagined", "visualized", "dreamt", "embodied", "spatialized", "timed"]
SOUND_TEXTURE_WORDS = ["airy","alien","ambient","angular","animated","asymmetrical","atonal","atmospheric","aural","bassy","beating","binary","bizarre","bitcrushed","blaring","blended","blippy","blooming","boiling","boomy","bouncing","bright","brittle","broken","bubbly","buzzing","chaotic","chattering","chimey","choppy","chunky","clean","clipped","cluttered","coarse","complex","compressed","concrete","consonant","continuous","crackling","creamy","crisp","crunchy","crushed","crystalline","crystal-clear","damaged","dark","deep","delicate","dense","detuned","dirty","disjointed","dissonant","distorted","dreamy","drifting","dripping","droning","dry","dubby","dull","dynamic","eerie","elastic","electronic","emphatic","ethereal","expansive","faint","fat","faulty","filtered","fizzy","flat","fluctuating","fluttering","foggy","folded","fragmented","freaky","frenetic","fresh","frosty","frozen","full","fuzzy","ghostly","glassy","glitchy","glistening","grainy","granular","grating","grimy","gritty","growling","groovy","grungy","harsh","haunting","hazy","heavy","hissing","hollow","homophonic","humid","hypnotic","icy","imposing","inharmonic","intense","intermittent","irregular","jagged","jittery","juicy","kaleidoscopic","laced","layered","leaky","light","lo-fi","loud","loopy","lush","mechanical","melodic","metallic","meandering","modulated","monochrome","monophonic","moody","morphing","moshable","muddy","murky","muted","mysterious","nautical","nebulous","noise-rich","noisy","nonlinear","nostalgic","obscured","offbeat","oscillating","overdriven","overlapping","overpowering","pad-like","percussive","phased","phat","phasing","plastic","plucky","pointillistic","popping","powerful","processed","pulsating","pure","pushy","quaking","quiet","raucous","raw","razored","resonant","resampled","reverberant","rich","ringing","roaring","robotic","rough","rumbling","runny","sandy","saturated","sawtoothy","scattered","sculpted","sharp","shelled","shimmery","shiny","short","sibilant","signal-like","silky","sizzling","skittering","slippery","sludgy","smoky","smooth","snappy","soft","solid","sonic","soothing","spacey","spectral","spiky","spongy","sprinkled","squelchy","static","stereo","stretched","subtle","surging","swarming","sweeping","sweet","syncopated","syrupy","taut","tense","textured","thick","thin","thready","throbbing","tidal","tight","tinny","torn","trembling","tremolo","tribal","trippy","turbulent","twangy","twinkly","uneven","unfiltered","unstable","untamed","velvety","vibrant","vibrating","vicious","vinyl","viscous","washed","warm","wavy","wet","whiny","whirring","whispery","wide","wispy","woody","wrinkled","wub-wub"]
EMOTIONAL_FEEL_WORDS = ["happy", "sad", "joyful", "angry", "hopeful", "hopeless", "anxious", "calm", "excited", "bored", "content", "distressed", "peaceful", "irritated", "elated", "miserable", "serene", "tense", "relaxed", "fearful", "brave", "melancholic", "cheerful", "lonely", "loving", "hateful", "grateful", "resentful", "proud", "ashamed", "confident", "insecure", "jealous", "empathetic", "apathetic", "passionate", "numb", "energetic", "weary", "overwhelmed", "inspired", "discouraged", "affectionate", "bitter", "forgiving", "vengeful", "trusting", "suspicious", "compassionate", "cruel", "sentimental", "indifferent", "nostalgic", "cynical", "optimistic", "pessimistic", "regretful", "remorseful", "triumphant", "defeated", "amused", "serious", "nervous", "secure", "panicked", "composed", "stunned", "surprised", "shocked", "pleased", "disappointed", "enraged", "agitated", "aggressive", "defensive", "guilty", "innocent", "vulnerable", "strong", "weak", "empowered", "powerless", "satisfied", "unsatisfied", "delighted", "heartbroken", "warm", "cold", "open", "closed", "breezy", "dreamy", "alert", "sluggish", "irritable", "euphoric", "devastated", "curious", "disinterested", "ecstatic", "mournful", "restless", "lazy", "determined", "uncertain", "decisive", "disillusioned", "motivated", "distracted", "present", "absent", "longing", "fulfilled", "desperate", "frustrated", "connected", "isolated", "inclusive", "exclusive", "harmonious", "conflicted", "caring", "detached", "cautious", "rigid", "sorrowful", "encouraged", "rejected", "loved", "unloved", "welcomed", "shunned", "celebrated", "ignored", "noticed", "overlooked", "respected", "disrespected", "validated", "invalidated", "cherished", "forgotten", "acknowledged", "dismissed", "admired", "envied", "pitied", "honored", "disgraced", "gentle", "harsh", "bold", "shy", "daring", "hesitant", "alive", "deadened", "radiant", "dull", "spontaneous", "calculated", "supportive", "undermined", "uplifted", "dragged", "enriched", "depleted", "refreshed", "drained", "balanced", "unstable", "centered", "scattered", "magnetic", "repelled", "glowing", "grim", "tender", "abrasive", "humble", "arrogant", "generous", "selfish", "open-hearted", "guarded", "burdened", "frantic", "stormy", "grounded", "playful", "stoic", "flirty", "intense", "mellow", "vibrant", "shadowed", "lit", "realistic", "floaty", "heavy", "bright", "somber", "jubilant", "dreary", "enthusiastic", "flat", "wild", "tame", "driven", "aimless", "zestful", "lively", "listless", "awake", "upbeat", "downcast", "thrilled", "yearning", "edgy", "aglow", "blank", "festive"]
SOUND_SOURCE_WORDS = ["oscillator","waveform","frequency","amplitude","tone","pitch","harmonics","timbre","resonance","noise","sine","square","triangle","sawtooth","pulse","wave","LFO","VCO","FM","AM","phase","sync","modulation","detune","overtone","subharmonic","spectral","signal","carrier","modulator","transient","attack","decay","sustain","release","envelope","ADSR","drone","hum","buzz","ring","chirp","blip","click","pop","glitch","grain","sample","loop","buffer","playback","oscillatorbank","additive","subtractive","granular","wavetable","phase-distortion","feedback","reverb","delay","echo","shimmer","flutter","rumble","hiss","static","impulse","burst","ping","chime","tonewheel","FM-operator","wavetable-index","formant","growl","bark","pluck","strike","bow","hit","scrape","tap","clang","bang","whisper","murmur","moan","shout","vocal","breath","noiseburst","pink-noise","white-noise","brown-noise","tone-generator","synth","synthesizer","analog","digital","hybrid","voice","patch","preset","program","timbral","expressive","tonal","atonal","microtonal","polyphonic","monophonic","stereo","mono","surround","binaural","3D-audio","spatial","localization","Doppler","ambience","source","emitter","listener","directional","omnidirectional","cardioid","figure-eight","field-recording","convolution","spectral-envelope","harmonic-series","resonator","exciter","physical-model","source-filter","bowing-model","string-model","brass-model","reed-model","voice-model","karplus-strong","PM","FM-index","modulation-depth","sync-ratio","harmonizer","unison","chorus","phase-shift","comb-filter","formant-shift","morph","blend","crossfade","pan","position","azimuth","elevation","directivity","soundfield","proximity","gain","level","mix","balance","dynamics","articulation","emphasis","inflection","timbral-shape","spectrum","bandwidth","filter","EQ","bandpass","lowpass","highpass","notch","shelving","cutoff","resonance-peak","Q-factor","modulation-source","control-voltage","envelope-follower","sequencer","gate","trigger","CV","MIDI","velocity","aftertouch","expression","controller","automation","touchpad","ribbon","joystick","XY-pad","modulation-wheel","pitch-bend","keyfollow","rate","depth","shape","curve","smoothing","lag","slew","noise-floor","bit-depth","sample-rate","aliasing","distortion","saturation","overdrive","fuzz","clip","wavefolder","wave-shaper","rectifier","quantizer","bitcrusher","vocoder","talkbox","resynthesis","spectral-freeze","freeze","stretch","time-domain","frequency-domain","analysis","synthesis","source-code","sound-object","audio-stream","tone-color","tone-quality","signature","fingerprint","character","texture","grainy","smooth","rough","warm","cold","bright","dark","rich","thin","full","hollow","metallic","wooden","glassy","rubbery","synthetic","natural","organic","electric","mechanical","digital","analogic","lo-fi","hi-fi"]
GENRE_STYLE_WORDS = ["Action","Adventure","Afrofuturism","Alternative","Americana","Anime","Art Deco","Art House","Art Punk","Avant-garde","Baroque","Beat","Big Band","Black Comedy","Blaxploitation","Blues","Bollywood","Boom Bap","Camp","Celtic","Chamber Pop","Chanson","Chillwave","Chiptune","Christian","Cinematic","Classic Rock","Classical","Cloud Rap","Comedy","Coming-of-age","Contemporary","Country","Crime","Crossover","Cult","Cyberpunk","Dark Ambient","Dark Fantasy","Darkwave","Death Metal","Deep House","Digital Hardcore","Disco","Docudrama","Documentary","Doom Metal","Downtempo","Dramedy","Drill","Drum and Bass","Dub","Dubstep","Dystopian","East Coast Rap","EDM","Electro","Electroclash","Electronic","Electropop","Emo","Emocore","Epic","Ethnic","Fairy Tale","Fantasy","Femme Fatale","Fiction","Film Noir","Folk","Folk Horror","Folk Punk","Folk Rock","Free Jazz","Freestyle","French House","Future Bass","Future Funk","Futurepop","Futurism","Game Show","Garage","Garage Rock","Giallo","Glam","Glam Rock","Glitch","Glitch Hop","Gospel","Goth","Gothic","Gothic Horror","Graffiti","Grime","Grindcore","Groove","Gypsy Jazz","Hard Rock","Hardcore","Hardcore Punk","Hardstyle","Hauntology","Heavy Metal","High Fantasy","Hip Hop","Historical","Holiday","Horror","House","Hyperpop","IDM","Impressionist","Indie","Indie Folk","Indie Pop","Indie Rock","Industrial","Instrumental","Intelligent Dance","International","Islamic","Jazz","Jazz Fusion","Jersey Club","Jungle","K-pop","Krautrock","Latin","Lo-fi","Lo-fi Hip Hop","Magical Realism","Martial Arts","Medieval","Melodic Death Metal","Melodrama","Metal","Metalcore","Midwest Emo","Minimalism","Mod","Modern Classical","Modernism","Motown","Mountain Music","Musical","Mystery","Mythology","Mythopoeia","Narrative","Nature","Nerdcore","New Age","New Wave","No Wave","Noise","Noise Rock","Noir","Nonfiction","Novelty","Nu Disco","Nu Metal","Nu-Jazz","Old School","Opera","Outlaw Country","Outsider","Parody","Period Drama","Phonk","Philosophical","Piano","Pixar-style","Plunderphonics","Poetry","Political","Pop","Pop Punk","Pop Rock","Post Hardcore","Post Punk","Post Rock","Post-apocalyptic","Postmodernism","Power Metal","Power Pop","Progressive","Progressive House","Progressive Metal","Progressive Rock","Psychedelic","Psychedelic Rock","Psychobilly","Punk","Punk Rock","Queercore","R&B","Ragga","Ragtime","Rap","Reggae","Reggaeton","Religious","Retro","Sci-fi","Screamo","Scripted","Shoegaze","Silent Film","Singer-Songwriter","Sitcom","Ska","Skate Punk","Slowcore","Smooth Jazz","Soft Rock","Soul","Sound Collage","Soundtrack","Space Rock","Spaghetti Western","Speculative","Spiritual","Spoken Word","Spy","Steampunk","Stoner Rock","Story-driven","Street Art","Streetwear","Subversive","Sufi","Synth-pop","Synthwave","Tarantinoesque","Tech House","Techno","Terrorcore","Thriller","Traditional","Tragedy","Trap","Tribal","Trip Hop","Twee","Underground","Urban","Utopian","Vaporwave","Variety Show","Victorian","Visual Novel","Visual Poetry","Vocal","Vocaloid","War","Western","Whodunit","Witch House","World","Worldbuilding","Zeitgeist","Zombie"]
SPATIAL_SENSE_WORDS = ["space","dimension","orientation","direction","angle","position","rotation","location","scale","proportion","distance","depth","height","width","length","volume","area","perimeter","boundary","coordinate","grid","latitude","longitude","elevation","azimuth","altitude","vector","trajectory","path","route","course","alignment","layout","geometry","symmetry","asymmetry","balance","form","shape","structure","pattern","topology","map","diagram","plan","schema","arrangement","organization","placement","distribution","spacing","proximity","closeness","separation","adjacency","contour","edge","border","margin","frame","context","environment","setting","zone","region","sector","plane","surface","layer","interface","intersection","overlap","juxtaposition","connection","link","relation","reference","origin","perspective","viewpoint","horizon","vanishing point","scale","zoom","compression","expansion","translation","transformation","reflection","rotation","mirroring","projection","distortion","foreshortening","relativity","parallax","gridline","map scale","topography","cartography","landscape","terrain","elevation model","contour line","navigation","compass","GPS","waypoint","landmark","beacon","spatial memory","mental map","cognitive map","wayfinding","tracking","motion","directionality","velocity","speed","acceleration","turn","bearing","heading","tilt","yaw","pitch","roll","movement","mobile","stationary","geolocation","localization","triangulation","trilateration","vector field","reference frame","coordinate system","XYZ axes","2D","3D","4D","spatial reasoning","spatial awareness","kinesthetic","proprioception","balance","sensing","perception","vision","visual cues","depth cues","stereo vision","binocular vision","monocular cues","occlusion","relative size","texture gradient","linear perspective","light and shadow","motion parallax","object permanence","egocentric","allocentric","spatial cognition","neural map","hippocampus","place cell","grid cell","spatial task","mental rotation","visualization","spatial language","preposition","above","below","under","over","beside","next to","between","inside","outside","near","far","behind","in front of","around","through","across","along","within","without","close","distant","nearby","adjacent","neighboring","continuous","discrete","modular","recursive","interconnected","interdependent","networked","clustered","isolated","compartmentalized","symmetrical","radial","axial","linear","nonlinear","organic","geometric","modular","repetitive","fractal","hierarchical","concentric","centralized","decentralized","distributed","zonal","segmented","oriented","scalable","navigable","tractable","measurable","quantifiable","bounded","unbounded","finite","infinite","relative","absolute","static","dynamic","stable","unstable","constrained","unconstrained","tactile","haptic","multisensory","interactive","immersive","virtual","augmented","real","imagined","abstract","concrete","conceptual","physical","mathematical"]
MOTION_CHARACTER_WORDS = ["fluid","dynamic","smooth","energetic","rhythmic","lively","graceful","agile","rapid","continuous","flowing","natural","elastic","flexible","subtle","bold","strong","sharp","swift","expressive","organic","spontaneous","lively","playful","dramatic","vibrant","intense","gentle","precise","seamless","naturalistic","exaggerated","controlled","unpredictable","quick","measured","slow","steady","jittery","bouncy","heavy","light","airy","weighty","deliberate","impulsive","evolving","recurring","looping","cascading","pulsing","wavering","trembling","oscillating","twitchy","swinging","rolling","sliding","gliding","swaying","rocking","flickering","fluttering","waving","spinning","twisting","turning","curling","stretching","compressing","contracting","expanding","bouncing","jittering","shuddering","shaking","trembling","rippling","shifting","warping","morphing","transforming","emerging","dissolving","fading","accelerating","decelerating","jerking","flicking","rolling","tumbling","balancing","jittering","flailing","bounding","darting","swooping","glancing","snapping","popping","rising","falling","sinking","soaring","diving","bursting","recoiling","collapsing","rebounding","wobbling","swerving","veering","cruising","creeping","slinking","slithering","twisting","writhing","wriggling","flowing","surging","ebbing","pulsing","vibrating","quivering","fluttering","staggering","stumbling","leaping","hopping","skipping","pacing","marching","stomping","striding","wandering","meandering","drifting","roaming","prowling","stalking","sneaking","crawling","slipping","skating","flying","flapping","hovering","spiraling","twirling","rotating","circling","swirling","undulating","shimmering","blazing","glowing","flashing","pulsating","beating","throbbing","pumping","streaming","rushing","roaring","howling","whirling","whizzing","zooming","zipping","skimming","skittering","toppling","flipping","cartwheeling","somersaulting","teetering","quaking","jolting","flinching","writhing","trembling","flickering","jittering","flowing","floating","sailing","cruising","meandering","prowling","slinking","slithering","crawling","skating","soaring","diving","swooping","spiraling","twirling","rotating","circling","swirling","oscillating","undulating","rippling","shimmering","sparkling","blazing","glowing","flashing","pulsating","beating","throbbing","pumping","surging","streaming","rushing","roaring","howling","whirling","whizzing","zooming","darting","skimming","skittering","sliding","tumbling","toppling","flipping","cartwheeling","somersaulting","balancing","teetering","swaying","quaking","shaking","jolting","jerking","flinching"]
CLARITY_QUALITY_WORDS = ["clear","precise","transparent","sharp","distinct","vivid","lucid","obvious","apparent","clean","pure","focused","accurate","unambiguous","simple","plain","straightforward","intelligible","explicit","neat","bright","crisp","flawless","excellent","perfect","superb","outstanding","premium","refined","pristine","impeccable","authentic","genuine","true","reliable","consistent","trustworthy","sound","solid","durable","robust","stable","polished","elegant","sophisticated","coherent","balanced","harmonious","smooth","seamless","articulate","well-defined","detailed","comprehensive","thorough","exhaustive","methodical","systematic","meticulous","careful","attentive","diligent","conscientious","efficient","effective","powerful","strong","resilient","sustainable","adaptable","innovative","creative","inspired","insightful","intelligent","smart","wise","knowledgeable","experienced","expert","skilled","talented","capable","competent","proficient","masterful","high-grade","top-notch","first-rate","state-of-the-art","advanced","cutting-edge","modern","refined","tasteful","cultured","stylish","chic","attractive","appealing","aesthetic","graceful","polished","impressive","distinguished","notable","remarkable","exceptional","noteworthy","admirable","praiseworthy","exemplary","ideal","optimal","ultimate","definitive","unmatched","unrivaled","supreme","superior","dominant","commanding","authoritative","prominent","leading","pioneering","trailblazing","groundbreaking","visionary","forward-thinking","proactive","responsive","attentive","engaging","communicative","approachable","transparent","open","honest","frank","straightforward","direct","clear-headed","rational","logical","analytical","critical","perceptive","observant","mindful","aware","focused","centered","calm","serene","peaceful","tranquil","composed","steady","balanced","harmonious","consistent","stable","dependable","resolute","persistent","determined","tenacious","unwavering","loyal","dedicated","committed","passionate","enthusiastic","energetic","dynamic","vibrant","lively","spirited","motivated","driven","ambitious","goal-oriented","organized","structured","planned","strategic","tactical","methodical","disciplined","rigorous","precise","exact","specific","measurable","tangible","practical","useful","functional","efficient","productive","streamlined","optimized","user-friendly","accessible","adaptable","flexible","versatile","scalable","maintainable","sustainable","environmentally-friendly","ethical","responsible","conscientious","transparent","accountable","fair","just","equitable","inclusive","diverse","respectful","courteous","polite","friendly","warm","inviting","welcoming","supportive","helpful","collaborative","cooperative","synergistic","innovative","creative","original","fresh","experimental","bold","daring","courageous","confident","assertive","persuasive","influential","charismatic","inspiring","motivating","uplifting","encouraging","positive","optimistic","hopeful","visionary","imaginative","inventive","resourceful","practical","grounded","realistic","pragmatic","sensible","reliable","consistent","stable","durable","resilient","robust","safe","secure","protected","trusted","verified","validated","confirmed","certified","guaranteed","endorsed","approved","compliant","standardized","regulated","licensed","insured","qualified","skilled","trained","educated","knowledgeable","experienced","expert","proficient","competent","capable","talented","gifted","masterful","professional","polished","elegant","refined","distinguished","notable","remarkable","exceptional","outstanding","superb","perfect","flawless","pristine","immaculate","faultless","exquisite","beautiful","attractive","appealing","tasteful","stylish","chic","fashionable","trendy","classic","timeless","enduring","legendary","iconic","prestigious","reputable","credible","trustworthy","honest","ethical","moral","virtuous","sincere","genuine","heartfelt","authentic"]
RHYTHMIC_FLOW_WORDS = ["beat","pulse","tempo","groove","cadence","rhythm","swing","timing","meter","syncopation","flow","pattern","motion","momentum","cycle","measure","accent","beatbox","bop","bounce","drum","drumbeat","drumroll","footstep","heartbeat","march","pace","phrase","rebound","repeat","roll","rythm","sequence","shuffle","snap","snare","soundwave","steady","strum","tap","tick","tick-tock","tone","track","triplet","twirl","vibration","wave","whip","waltz","whirl","zip","zoom","animate","balance","beatitude","breathe","cadence","call-and-response","chain","chant","chatter","chase","circuit","clutch","color","dance","dash","delivery","drift","echo","elevate","engage","expression","flare","flex","flowchart","flourish","flutter","fold","funk","glide","glissando","grace","groove","gyrate","harmonic","heartbeat","highlight","hook","hum","impulse","intensity","interplay","jive","jump","kick","kinetic","layer","leap","lengthen","lilt","loop","magic","measure","melody","meter","modulation","motion","move","movement","music","natural","nuance","offset","oscillate","pace","passage","pattern","pedal","perform","phrase","pitch","pulse","push","quicken","quote","race","rhythmics","ripple","roll","roll-off","rush","scan","score","segment","sequence","shake","shift","signal","slide","slip","snap","soar","sound","sparkle","spin","spiral","split","sprint","step","stretch","string","stroke","swing","swirl","synth","tempo","throb","thrust","tick","tone","trace","track","transfer","tremor","trip","tune","twang","twist","undulate","vibration","wave","whirl","wind","yawn","zip","zoom","agile","alive","animate","balance","bounce","brisk","cadence","cascade","chase","circuit","cluster","color","dart","dash","delay","descent","drift","drop","dynamic","elevate","energize","engage","flicker","flow","fluctuate","flux","fly","glide","grace","groove","gyrate","harmonic","hustle","impulse","jolt","jump","kinetic","lilt","loop","motion","move","pace","pass","pulse","push","quicken","race","ripple","roll","rush","scatter","score","sequence","shake","shift","slide","snap","spin","spiral","sprint","step","stretch","swirl","synth","tempo","throb","thrust","tick","tone","track","transfer","tremor","trip","tune","twist","undulate","vibration","whirl","wind","zip","zoom","zest","zeal","zipline","zing","zingy"]
MORE_EMOTION_WORDS = ["lethargy", "boredom", "serenity", "fatigue", "indifference", "calm", "dullness", "apathy", "relaxation", "sadness", "disappointment", "contentment", "melancholy", "uncertainty", "satisfaction", "grief", "resignation", "hopefulness", "despair", "confusion", "curiosity", "discouragement", "hesitation", "optimism", "fear", "alertness", "interest", "worry", "anticipation", "encouragement", "nervousness", "surprise", "amusement", "anxiety", "tension", "enjoyment", "frustration", "uncertainty", "pleasure", "anger", "irritation", "excitement", "hostility", "suspense", "happiness", "rage", "alarm", "joy", "panic", "conflicted", "delight", "terror", "dissonance", "enthusiasm", "shock", "surprise", "euphoria", "agitation", "stimulation", "ecstasy", "disgust", "concern", "love", "shame", "reflection", "affection", "guilt", "acceptance", "gratitude", "resentment", "contemplation", "appreciation", "envy", "puzzlement", "inspiration", "contempt", "awe", "compassion", "bitterness", "curiosity", "admiration"]
SYNTH_WORDS = ["airy ambience", "airy analog", "airy beat", "airy chime", "airy delay", "airy effect", "airy groove", "airy kick", "airy modulation", "airy piano", "airy pluck", "airy sequence", "airy shimmer", "airy square", "airy texture", "airy timbre", "airy vibe", "airy wave", "ambient ambience", "ambient bass", "ambient bassline", "ambient beat", "ambient brass", "ambient filter", "ambient groove", "ambient harmony", "ambient piano", "ambient plucks", "ambient pulse", "ambient saw", "ambient swell", "ambient tone", "ambient vibe", "ambient voice", "ambient wash", "ambient wave", "bitcrushed analog", "bitcrushed arpeggio", "bitcrushed beat", "bitcrushed bells", "bitcrushed brass", "bitcrushed digital", "bitcrushed groove", "bitcrushed harmony", "bitcrushed modulation", "bitcrushed pad", "bitcrushed pulse", "bitcrushed reverb", "bitcrushed square", "bitcrushed strings", "bitcrushed texture", "bright ambience", "bright chime", "bright effect", "bright groove", "bright harmony", "bright lead", "bright loop", "bright modulation", "bright oscillator", "bright pluck", "bright plucks", "bright rhythm", "bright sequence", "bright shimmer", "bright sweep", "bright texture", "bright tone", "bright wash", "bright wave", "bubbly ambience", "bubbly analog", "bubbly chime", "bubbly delay", "bubbly digital", "bubbly effect", "bubbly groove", "bubbly lead", "bubbly loop", "bubbly modulation", "bubbly motion", "bubbly noise", "bubbly piano", "bubbly pluck", "bubbly sequence", "bubbly square", "bubbly strings", "bubbly swell", "bubbly wash", "chirpy ambience", "chirpy atmosphere", "chirpy brass", "chirpy chords", "chirpy delay", "chirpy digital", "chirpy filter", "chirpy groove", "chirpy loop", "chirpy melody", "chirpy piano", "chirpy pulse", "chirpy saw", "chirpy sequence", "chirpy square", "chirpy vibe", "chirpy voice", "cinematic ambience", "cinematic arpeggio", "cinematic atmosphere", "cinematic bassline", "cinematic beat", "cinematic chime", "cinematic chords", "cinematic delay", "cinematic effect", "cinematic filter", "cinematic loop", "cinematic modulation", "cinematic motion", "cinematic pad", "cinematic plucks", "cinematic pulse", "cinematic reverb", "cinematic square", "cinematic strings", "cinematic sweep", "cinematic swell", "cinematic timbre", "cinematic tone", "cinematic voice", "crunchy ambience", "crunchy analog", "crunchy arpeggio", "crunchy atmosphere", "crunchy bass", "crunchy beat", "crunchy digital", "crunchy drone", "crunchy echo", "crunchy effect", "crunchy filter", "crunchy kick", "crunchy loop", "crunchy motion", "crunchy oscillator", "crunchy ring", "crunchy saw", "crunchy sequence", "crunchy square", "crunchy texture", "crunchy tone", "crunchy voice", "crunchy wash", "dark analog", "dark beat", "dark chime", "dark chords", "dark drone", "dark echo", "dark effect", "dark filter", "dark harmony", "dark melody", "dark modulation", "dark motion", "dark noise", "dark oscillator", "dark reverb", "dark rhythm", "dark ring", "dark sequence", "dark square", "dark strings", "dark vibe", "dark voice", "dark wave", "deep atmosphere", "deep bass", "deep beat", "deep bells", "deep brass", "deep chime", "deep delay", "deep echo", "deep effect", "deep groove", "deep loop", "deep melody", "deep modulation", "deep motion", "deep piano", "deep sequence", "deep square", "deep sweep", "deep texture", "deep timbre", "deep vibe", "detuned ambience", "detuned chime", "detuned chords", "detuned filter", "detuned kick", "detuned lead", "detuned modulation", "detuned noise", "detuned pad", "detuned pulse", "detuned reverb", "detuned rhythm", "detuned saw", "detuned shimmer", "detuned vibe", "detuned wave", "distorted bassline", "distorted bells", "distorted brass", "distorted delay", "distorted effect", "distorted modulation", "distorted motion", "distorted noise", "distorted oscillator", "distorted piano", "distorted plucks", "distorted pulse", "distorted rhythm", "distorted sequence", "distorted strings", "distorted texture", "distorted tone", "distorted voice", "dreamy atmosphere", "dreamy bassline", "dreamy bells", "dreamy brass", "dreamy chime", "dreamy drone", "dreamy effect", "dreamy filter", "dreamy groove", "dreamy harmony", "dreamy lead", "dreamy loop", "dreamy melody", "dreamy oscillator", "dreamy pad", "dreamy pulse", "dreamy rhythm", "dreamy ring", "dreamy saw", "dreamy sweep", "dreamy swell", "dreamy timbre", "dreamy tone", "dreamy voice", "ethereal analog", "ethereal bass", "ethereal bassline", "ethereal chime", "ethereal chords", "ethereal digital", "ethereal echo", "ethereal effect", "ethereal filter", "ethereal groove", "ethereal harmony", "ethereal lead", "ethereal loop", "ethereal noise", "ethereal pad", "ethereal pluck", "ethereal pulse", "ethereal tone", "evolving ambience", "evolving analog", "evolving bassline", "evolving chime", "evolving chords", "evolving filter", "evolving groove", "evolving harmony", "evolving loop", "evolving oscillator", "evolving plucks", "evolving reverb", "evolving rhythm", "evolving square", "evolving strings", "evolving vibe", "fat ambience", "fat atmosphere", "fat bassline", "fat bells", "fat brass", "fat filter", "fat groove", "fat kick", "fat pad", "fat piano", "fat pulse", "fat sequence", "fat strings", "fat sweep", "fat swell", "fat texture", "fat timbre", "fat vibe", "fat wash", "fuzzy ambience", "fuzzy analog", "fuzzy arpeggio", "fuzzy chords", "fuzzy delay", "fuzzy digital", "fuzzy echo", "fuzzy groove", "fuzzy kick", "fuzzy melody", "fuzzy saw", "fuzzy square", "fuzzy texture", "fuzzy tone", "fuzzy wash", "fuzzy wave", "ghostly ambience", "ghostly arpeggio", "ghostly bassline", "ghostly beat", "ghostly brass", "ghostly drone", "ghostly groove", "ghostly harmony", "ghostly modulation", "ghostly motion", "ghostly noise", "ghostly oscillator", "ghostly pad", "ghostly piano", "ghostly pluck","ghostly plucks", "ghostly saw", "ghostly wash", "glassy ambience", "glassy arpeggio", "glassy atmosphere", "glassy beat", "glassy chime", "glassy delay", "glassy drone", "glassy filter", "glassy kick", "glassy lead", "glassy noise", "glassy piano", "glassy pluck", "glassy saw", "glassy sequence", "glassy sweep", "glassy vibe", "glassy wash", "glassy wave", "glitchy ambience", "glitchy atmosphere", "glitchy bass", "glitchy bassline", "glitchy brass", "glitchy chime", "glitchy chords", "glitchy delay", "glitchy digital", "glitchy drone", "glitchy filter", "glitchy kick", "glitchy lead", "glitchy loop", "glitchy melody", "glitchy oscillator", "glitchy pluck","glitchy plucks", "glitchy reverb", "glitchy rhythm", "glitchy saw", "glitchy sequence", "glitchy sweep", "glitchy tone", "glitchy vibe", "glitchy voice", "glitchy wash", "granular ambience", "granular analog", "granular arpeggio", "granular bass", "granular beat", "granular bells", "granular chime", "granular delay", "granular digital", "granular groove", "granular kick", "granular lead", "granular loop", "granular motion", "granular plucks", "granular reverb", "granular strings", "granular voice", "granular wave", "gritty ambience", "gritty arpeggio", "gritty atmosphere", "gritty bassline", "gritty bells", "gritty brass", "gritty delay", "gritty groove", "gritty harmony", "gritty kick", "gritty motion", "gritty pad", "gritty piano", "gritty sequence", "gritty shimmer", "gritty square", "gritty swell", "gritty tone", "gritty wave", "harsh ambience", "harsh bass", "harsh beat", "harsh chime", "harsh chords", "harsh drone", "harsh echo", "harsh lead", "harsh noise", "harsh pad", "harsh pulse", "harsh reverb", "harsh ring", "harsh saw", "harsh square", "harsh sweep", "harsh timbre", "harsh tone", "harsh wash", "icy analog", "icy atmosphere", "icy bass", "icy bassline", "icy beat", "icy chime", "icy echo", "icy filter", "icy oscillator", "icy piano", "icy pluck", "icy plucks", "icy rhythm", "icy ring", "icy saw", "icy sequence", "icy square", "icy texture", "icy vibe", "icy voice", "icy wash", "icy wave", "lo-fi ambience", "lo-fi analog", "lo-fi bass", "lo-fi beat", "lo-fi delay", "lo-fi groove", "lo-fi lead", "lo-fi loop", "lo-fi modulation", "lo-fi motion", "lo-fi pad", "lo-fi piano", "lo-fi pluck", "lo-fi pulse", "lo-fi reverb", "lo-fi rhythm", "lo-fi sweep", "lo-fi texture", "lo-fi timbre", "lo-fi tone", "lo-fi vibe", "lo-fi voice", "lo-fi wash", "lush atmosphere", "lush bells", "lush chords", "lush delay", "lush digital", "lush effect", "lush groove", "lush kick", "lush plucks", "lush pulse", "lush reverb", "lush rhythm", "lush ring", "lush sequence", "lush voice", "lush wave", "mechanical analog", "mechanical bass", "mechanical beat", "mechanical brass", "mechanical delay", "mechanical digital", "mechanical echo", "mechanical groove", "mechanical loop", "mechanical motion", "mechanical pluck", "mechanical rhythm", "mechanical timbre", "mechanical voice", "mechanical wash", "mellow atmosphere", "mellow bassline", "mellow bells", "mellow brass", "mellow digital", "mellow drone", "mellow filter", "mellow groove", "mellow melody", "mellow plucks", "mellow reverb", "mellow rhythm", "mellow ring", "mellow saw", "mellow shimmer", "mellow square", "mellow tone", "mellow voice", "metallic bassline", "metallic chords", "metallic delay", "metallic effect", "metallic harmony", "metallic motion", "metallic noise", "metallic oscillator", "metallic pad", "metallic plucks", "metallic rhythm", "metallic shimmer", "metallic square", "metallic sweep", "metallic swell", "metallic timbre", "metallic vibe", "metallic wave", "modulated atmosphere", "modulated bassline", "modulated beat", "modulated brass", "modulated chime", "modulated delay", "modulated digital", "modulated echo", "modulated effect", "modulated filter", "modulated groove", "modulated kick", "modulated lead", "modulated pluck", "modulated plucks", "modulated rhythm", "modulated ring", "modulated square", "modulated strings", "modulated swell", "modulated timbre", "modulated voice", "modulated wash", "noisy ambience", "noisy bassline", "noisy beat", "noisy delay", "noisy digital", "noisy harmony", "noisy lead", "noisy loop", "noisy noise", "noisy pluck", "noisy plucks", "noisy sequence", "noisy shimmer", "noisy square", "noisy swell", "noisy tone", "noisy vibe", "noisy wash", "noisy wave", "organic arpeggio", "organic atmosphere", "organic bass", "organic bassline", "organic digital", "organic echo", "organic effect", "organic groove", "organic kick", "organic lead", "organic melody", "organic motion", "organic noise", "organic pluck", "organic plucks", "organic reverb", "organic ring", "organic sequence", "organic sweep", "organic texture", "organic tone", "organic voice", "psychedelic arpeggio", "psychedelic bassline", "psychedelic brass", "psychedelic chords", "psychedelic digital", "psychedelic echo", "psychedelic kick", "psychedelic melody", "psychedelic modulation", "psychedelic motion", "psychedelic noise", "psychedelic pad", "psychedelic pluck", "psychedelic pulse", "psychedelic rhythm", "psychedelic ring", "psychedelic square", "psychedelic texture", "psychedelic tone", "punchy ambience", "punchy bass", "punchy bells", "punchy chords", "punchy echo", "punchy filter", "punchy groove", "punchy oscillator", "punchy pluck", "punchy pulse", "punchy swell", "punchy texture", "punchy timbre", "punchy vibe", "punchy wash", "resonant bassline", "resonant brass", "resonant effect", "resonant groove", "resonant kick", "resonant loop", "resonant motion", "resonant noise", "resonant oscillator", "resonant pad", "resonant pulse", "resonant ring", "resonant shimmer", "resonant sweep", "resonant texture", "resonant vibe", "resonant voice", "resonant wave", "retro analog", "retro arpeggio", "retro bass", "retro delay", "retro drone", "retro filter", "retro groove", "retro kick", "retro melody", "retro modulation", "retro noise", "retro piano", "retro reverb", "retro timbre", "retro vibe", "robotic bass", "robotic bells", "robotic chime", "robotic chords", "robotic digital", "robotic drone", "robotic effect", "robotic filter", "robotic groove", "robotic harmony", "robotic lead", "robotic melody", "robotic oscillator", "robotic pad", "robotic pulse", "robotic ring", "robotic sequence", "robotic shimmer", "robotic sweep", "robotic vibe", "rubbery arpeggio", "rubbery atmosphere", "rubbery brass", "rubbery chime", "rubbery chords", "rubbery delay", "rubbery echo", "rubbery filter", "rubbery groove", "rubbery harmony", "rubbery lead", "rubbery pad", "rubbery plucks", "rubbery sequence", "rubbery shimmer", "rubbery square", "rubbery sweep", "rubbery swell", "rubbery voice", "rubbery wash", "scary ambience", "scary arpeggio", "scary bass", "scary beat", "scary chords", "scary delay", "scary drone", "scary effect", "scary groove", "scary lead", "scary modulation", "scary noise", "scary plucks", "scary ring", "scary saw", "scary shimmer", "scary square", "scary texture", "scary timbre", "scary tone", "scary wash", "sharp beat", "sharp chime", "sharp digital", "sharp effect", "sharp groove", "sharp kick", "sharp lead", "sharp modulation", "sharp noise", "sharp piano", "sharp reverb", "sharp sequence", "sharp tone", "sharp voice", "sharp wash", "shimmering ambience", "shimmering analog", "shimmering atmosphere", "shimmering bassline", "shimmering beat", "shimmering bells", "shimmering delay", "shimmering effect", "shimmering filter", "shimmering harmony", "shimmering lead", "shimmering loop", "shimmering melody", "shimmering modulation", "shimmering motion", "shimmering oscillator", "shimmering pad", "shimmering pluck", "shimmering plucks", "shimmering reverb", "shimmering ring", "shimmering sequence", "shimmering sweep", "shimmering swell", "shimmering texture", "shimmering vibe", "shimmering wash", "smooth bass", "smooth bassline", "smooth beat", "smooth digital", "smooth drone", "smooth effect", "smooth filter", "smooth groove", "smooth kick", "smooth loop", "smooth motion", "smooth noise", "smooth piano", "smooth pulse", "smooth reverb", "smooth rhythm", "smooth ring", "smooth saw", "smooth shimmer", "smooth square", "smooth strings", "smooth swell", "smooth texture", "smooth tone", "smooth voice", "smooth wave", "squelchy ambience", "squelchy analog", "squelchy atmosphere", "squelchy beat", "squelchy brass", "squelchy effect", "squelchy filter", "squelchy lead", "squelchy loop", "squelchy motion", "squelchy noise", "squelchy oscillator", "squelchy pad", "squelchy piano", "squelchy plucks", "squelchy pulse", "squelchy ring", "squelchy saw", "squelchy sequence", "squelchy sweep", "squelchy tone", "squelchy wash", "subtle atmosphere", "subtle bells", "subtle brass", "subtle chords", "subtle drone", "subtle filter", "subtle harmony", "subtle kick", "subtle loop", "subtle melody", "subtle oscillator", "subtle pad", "subtle plucks", "subtle ring", "subtle saw", "subtle shimmer", "subtle strings", "subtle sweep", "subtle timbre", "subtle tone", "subtle wave", "synthetic analog", "synthetic atmosphere", "synthetic beat", "synthetic bells", "synthetic chime", "synthetic drone", "synthetic echo", "synthetic filter", "synthetic harmony", "synthetic lead", "synthetic melody", "synthetic oscillator", "synthetic pad", "synthetic piano", "synthetic pluck", "synthetic plucks", "synthetic pulse", "synthetic reverb", "synthetic rhythm", "synthetic ring", "synthetic saw", "synthetic strings", "synthetic sweep", "synthetic texture", "synthetic vibe", "synthetic voice", "twinkly ambience", "twinkly analog", "twinkly bassline", "twinkly beat", "twinkly bells", "twinkly brass", "twinkly chords", "twinkly delay", "twinkly effect", "twinkly groove", "twinkly harmony", "twinkly loop", "twinkly melody", "twinkly noise", "twinkly oscillator", "twinkly pad", "twinkly piano", "twinkly pluck", "twinkly plucks", "twinkly reverb", "twinkly saw", "twinkly sequence", "twinkly sweep", "twinkly tone", "twinkly voice", "twinkly wave", "twisted ambience", "twisted analog", "twisted arpeggio", "twisted atmosphere", "twisted beat", "twisted bells", "twisted brass", "twisted chime", "twisted delay", "twisted drone", "twisted filter", "twisted groove", "twisted lead", "twisted modulation", "twisted motion", "twisted pad", "twisted plucks", "twisted reverb", "twisted rhythm", "twisted ring", "twisted sweep", "twisted vibe", "twisted voice", "twisted wave", "vaporwave atmosphere", "vaporwave beat", "vaporwave digital", "vaporwave drone", "vaporwave effect", "vaporwave filter", "vaporwave harmony", "vaporwave kick", "vaporwave lead", "vaporwave loop", "vaporwave melody", "vaporwave modulation", "vaporwave pad", "vaporwave piano", "vaporwave plucks", "vaporwave rhythm", "vaporwave ring", "vaporwave saw", "vaporwave strings", "vaporwave sweep", "vaporwave texture", "vaporwave voice", "vibrant analog", "vibrant arpeggio", "vibrant atmosphere", "vibrant bass", "vibrant bells", "vibrant brass", "vibrant chime", "vibrant chords", "vibrant drone", "vibrant echo", "vibrant effect", "vibrant harmony", "vibrant lead", "vibrant melody", "vibrant modulation", "vibrant motion", "vibrant noise", "vibrant oscillator", "vibrant saw", "vibrant sequence", "vibrant square", "vibrant sweep", "vibrant texture", "vibrant vibe", "warm bassline", "warm delay", "warm digital", "warm effect", "warm groove", "warm kick", "warm loop", "warm modulation", "warm noise", "warm oscillator", "warm pluck", "warm rhythm", "warm saw", "warm shimmer", "warm square", "warm texture", "warm timbre", "warm vibe"]
WORD_LISTS = [
    TYPE_WORDS, NOISE_WORDS, INSTRUMENT_WORDS, 
    SYNTH_WORDS, TONE_QUALITY_WORDS, PITCH_LEVEL_WORDS, VOLUME_INTENSITY_WORDS, TIME_SHAPE_WORDS, SOUND_TEXTURE_WORDS, EMOTIONAL_FEEL_WORDS, MORE_EMOTION_WORDS, SOUND_SOURCE_WORDS, GENRE_STYLE_WORDS, SPATIAL_SENSE_WORDS, MOTION_CHARACTER_WORDS, CLARITY_QUALITY_WORDS, RHYTHMIC_FLOW_WORDS
]

def get_categories():
    """Return list of category names in order"""
    return [
        "Type", "Noise", "Instrument", "Synth", "Tone Quality", 
        "Pitch Level", "Volume Intensity", "Time Shape", "Sound Texture",
        "Emotional Feel", "More Emotion", "Sound Source", "Genre Style",
        "Spatial Sense", "Motion Character", "Clarity Quality", "Rhythmic Flow"
    ]

def parse_user_prompt(prompt: str) -> str:
    """Convert user prompt to comma-separated category values"""
    categories = get_categories()
    words = prompt.split()
    result = []
    
    # Map words to categories or use defaults
    for cat in categories:
        if len(words) > 0:
            result.append(words.pop(0))
        else:
            result.append("neutral")  # Default value
            
    return ",".join(result)

# Setup logging at the start of the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_classification.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Config:
    audio_dir: str = r"C:\Users\stuart\Documents\python_projects\ouput_midi_input_audo"
    params_csv: str = r"C:\Users\stuart\Documents\python_projects\ouput_midi_input_audo\restricted_parameter_data.csv"
    output_csv: str = r"C:\Users\stuart\Documents\python_projects\ouput_midi_input_audo\audio_analysis.csv"
    model_name: str = "laion/clap-htsat-unfused"
    llm_api_key: str = "lm-studio"
    llm_base_url: str = "http://localhost:1234/v1"
    min_matches: int = 4
    max_additional: int = 50
    sample_rate: int = 48000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k: int = 5  # Add missing config value
    log_file: str = 'audio_classification.log'
    
class AudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_model()
        self.processor = self._load_processor()
        
    def _load_model(self) -> ClapModel:
        """Load and configure the CLAP model"""
        try:
            model = ClapModel.from_pretrained(self.config.model_name)
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
            
    def _load_processor(self) -> AutoProcessor:
        """Load and configure the processor"""
        try:
            return AutoProcessor.from_pretrained(self.config.model_name)
        except Exception as e:
            logging.error(f"Failed to load processor: {str(e)}")
            raise

    def load_and_preprocess_audio(self, filepath: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sample_rate = torchaudio.load(filepath)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Move waveform to device
        waveform = waveform.to(self.device)
        
        # Resample to 48kHz
        if sample_rate != self.config.sample_rate:
            resampler = T.Resample(sample_rate, self.config.sample_rate)
            resampler = resampler.to(self.device)
            waveform = resampler(waveform)
            
        return waveform

    @torch.no_grad()  # Add decorator for inference
    def find_matching_word(self, filepath: str, word_list: List[str]) -> str:
        """Find best matching word for audio file"""
        try:
            waveform = self.load_and_preprocess_audio(filepath)
            audio_input = waveform.cpu().squeeze().numpy()
            
            inputs = self._prepare_inputs(audio_input, word_list)
            probs = self._get_predictions(inputs)
            
            return self._get_top_matches(probs, word_list)
        except Exception as e:
            logging.error(f"Error processing {filepath}: {str(e)}")
            raise

    def _prepare_inputs(self, audio_input: np.ndarray, word_list: List[str]) -> Dict:
        """Prepare model inputs"""
        inputs = self.processor(
            text=word_list,
            audios=[audio_input],
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True
        )
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}

    def _get_predictions(self, inputs: Dict) -> torch.Tensor:
        """Get model predictions"""
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs)
            return outputs.logits_per_audio.softmax(dim=-1)

    def _get_top_matches(self, probs: torch.Tensor, word_list: List[str]) -> str:
        """Get top matching words"""
        _, top_idx = torch.topk(probs[0], k=1)
        return word_list[top_idx[0]]

    def process_audio_files(self):
        """Process all audio files in directory and generate classifications"""
        # Read parameters from CSV
        params_dict = process_csv_files(self.config)
        
        # Create/check output CSV file
        if not os.path.exists(self.config.output_csv):
            with open(self.config.output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', 'Parameters', 'Generated_String'])
        
        # Get list of audio files
        audio_files = [f for f in os.listdir(self.config.audio_dir) if f.endswith(".wav")]
        
        # Process each WAV file with progress bar
        for filename in tqdm(audio_files, desc="Processing audio files"):
            try:
                # Get index from filename
                index = filename.replace(".wav", "")
                
                # Get parameters for this index
                if index not in params_dict:
                    continue  # Skip silently
                    
                params = params_dict[index]
                
                # Process the audio file
                filepath = os.path.join(self.config.audio_dir, filename)
                
                word_list = []
                for category_idx, word_list_category in enumerate(tqdm(WORD_LISTS, desc="Categories", leave=False)):
                    word = self.find_matching_word(filepath, word_list_category)
                    word_list.append(word)
                
                generated_string = ",".join(word_list)
                
                # Append to CSV file
                with open(self.config.output_csv, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([index, json.dumps(params), generated_string])
                    csvfile.flush()
                    
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

class MatchFinder:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=None
        )

    def find_best_matches(self, result: str, csv_file: str) -> Optional[str]:
        """Find best matching rows and use LLM to choose final match"""
        categories = self._get_categories()
        search_words = self._parse_search_words(result)
        added_words = {cat: set() for cat in categories}
        
        with tqdm(total=self.config.max_additional, desc="Expanding search") as pbar:
            matches = self._find_matches_iterative(
                csv_file, 
                search_words, 
                added_words,
                pbar
            )
        
        if not matches:
            logging.info("No matches found")
            return None
            
        best_matches = self._get_best_matches(matches)
        
        if len(best_matches) == 1:
            return best_matches[0][0]
            
        return self._select_best_match_llm(best_matches, search_words, added_words)

    def _get_similar_word(self, word: str, category: str, used_words: set) -> Optional[str]:
        """Get similar word using LLM"""
        category_list = [w.lower() for w in WORD_LISTS[self._get_category_index(category)]]
        available_words = [w for w in category_list if w not in used_words | {word.lower()}]
        
        if not available_words:
            return None
            
        prompt = self._build_similar_word_prompt(word, category, available_words, used_words)
        
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.2-1b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return completion.choices[0].message.content.strip().lower()
        except Exception as e:
            logging.error(f"LLM error: {str(e)}")
            return None

    def _find_matches_iterative(
        self, 
        csv_file: str, 
        search_words: Dict[str, str],
        added_words: Dict[str, set],
        pbar: tqdm
    ) -> Dict[str, Dict]:
        """Find matches iteratively, expanding search terms"""
        matches = {}
        attempts = 0
        
        while attempts < self.config.max_additional:
            pbar.update(1)
            
            if attempts > 0:
                if not self._expand_search_terms(search_words, added_words):
                    break;
                    
            current_matches = self._check_csv_matches(
                csv_file, 
                search_words,
                added_words
            )
            
            if current_matches:
                matches = current_matches
                break
                
            attempts += 1
            
        return matches

    def _select_best_match_llm(
        self, 
        matches: List[Tuple[str, Dict]], 
        search_words: Dict[str, str],
        added_words: Dict[str, set]
    ) -> Optional[str]:
        """Use LLM to select best match from candidates"""
        for attempt in range(3):
            prompt = self._build_selection_prompt(matches, search_words, added_words)
            
            try:
                completion = self.client.chat.completions.create(
                    model="llama-3.2-1b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                response = completion.choices[0].message.content.strip()
                
                if response in [str(index) for index, _ in matches]:
                    return response
                    
                logging.warning(f"Invalid LLM response on attempt {attempt + 1}")
            except Exception as e:
                logging.error(f"LLM error: {str(e)}")
                
        return None
def find_best_matches_with_llm(result, csv_file, min_matches=4, max_additional=50):
    """Find best matching rows and use LLM to choose final match, expanding search when needed"""
    import csv
    from openai import OpenAI
    import random
    
    # Initialize OpenAI client with logging suppression
    client = OpenAI(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        timeout=None,
        max_retries=0,  # Reduce retry noise
        default_headers={"user-agent": "audio-classifier/1.0"}
    )
    
    def get_similar_word(word, category, category_list):
        """Use LLM to find similar word from list, ensuring lowercase comparison"""
        # Convert category list and word to lowercase
        category_list = [w.lower() for w in category_list]
        word = word.lower()
        
        # Exclude words already used
        available_words = [w for w in category_list 
                        if w not in {word} | added_words[category]]
        
        if not available_words:
            return None
            
        prompt = f"""Given the following word{"s" if len(added_words) > len(category_list) else ""} "{word}{"," if len(added_words) > len(category_list) else ""}{', '.join(added_words[category])}" in category "{category}",
        find another similar word from this list: {','.join(available_words)}
        Always respond with a word from the list. Don't respond with any other text."""
        
        completion = client.chat.completions.create(
            model="llama-3.2-1b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return completion.choices[0].message.content.strip().lower()
    
    # Initialize categories and search words
    categories = get_categories()
    search_words = {cat: word.lower().strip() 
                   for cat, word in zip(categories, result.split(','))}
    original_words = search_words.copy()
    
    # Track added words by category
    added_words = {cat: set() for cat in categories}
    matches = {}
    attempts = 0
    
    # Main progress bar for word expansion
    with tqdm(total=max_additional, desc="Expanding search", unit="attempts") as pbar:
        while attempts < max_additional:
            # Add similar words for all categories
            if attempts > 0:
                words_added = False
                # Progress bar for categories
                for category in tqdm(categories, desc="Finding similar words", leave=False):
                    original_word = original_words[category]
                    category_idx = categories.index(category)
                    category_list = WORD_LISTS[category_idx]
                    
                    similar_word = get_similar_word(original_word, category, category_list)
                    
                    if similar_word and similar_word not in added_words[category]:
                        added_words[category].add(similar_word)
                        words_added = True
                
                if not words_added:
                    break
            
            # Check matches after adding new words
            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                matches.clear()
                
                # Progress bar for checking matches
                for row in tqdm(rows, desc="Checking matches", leave=False):
                    # Convert CSV row words to lowercase
                    row_words = {cat: word.lower().strip() 
                               for cat, word in zip(categories, row['Generated_String'].split(','))}
                    
                    # Count matches including original and added words
                    match_count = sum(1 for cat in categories 
                                    if row_words[cat] in 
                                    ({original_words[cat]} | added_words[cat]))
                    
                    if match_count >= min_matches:
                        matches[row['Index']] = {
                            'count': match_count,
                            'words': row_words,
                            'generated_string': row['Generated_String']
                        }
            
            # Check if we found matches meeting minimum criteria
            if matches:
                best_count = max(match['count'] for match in matches.values())
                best_matches = [(index, data) for index, data in matches.items() 
                               if data['count'] == best_count]
                pbar.set_postfix({'matches': len(best_matches), 'best_count': best_count})
                break
            
            attempts += 1
            pbar.update(1)
    
    if not matches:
        logging.info(f"No matches found with {min_matches}+ matching words")
        return None
    
    if len(best_matches) == 1:
        return best_matches[0][0]
    
    # Use LLM to choose between multiple matches
    for attempt in tqdm(range(3), desc="LLM selection", leave=False):
        prompt = f"""Original categories:\n"""
        for cat in categories:
            prompt += f"{cat}: {original_words[cat]}\n"
        
        prompt += "\nAdded similar words:\n"
        for cat in categories:
            if added_words[cat]:
                prompt += f"{cat}: {', '.join(added_words[cat])}\n"
        
        prompt += "\nCompare to these samples:\n"
        random_matches = best_matches.copy()
        random.shuffle(random_matches)
        
        for index, data in random_matches:
            words = data['words']
            prompt += f"\nSample {index}:\n"
            for cat in categories:
                prompt += f"{cat}: {words[cat]}\n"
        
        prompt += "\nWhich sample index best matches the original categories? Respond only with the index number."
        
        completion = client.chat.completions.create(
            model="llama-3.2-1b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        response = completion.choices[0].message.content.strip()
        
        if response in [str(index) for index, _ in best_matches]:
            return response
    
    return None

def process_csv_files(config: Config):
    """Process input and output CSV files with proper resource handling"""
    params_dict = {}
    
    with open(config.params_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        
        for row in tqdm(reader, desc="Processing CSV"):
            try:
                # Use positional indices like the original code
                index = row[0]
                # Skip timestamp in row[1]
                params_json = json.loads(row[2])  # Parameters in third column
                params_dict[index] = params_json
            except (json.JSONDecodeError, IndexError) as e:
                logging.error(f"Error processing row: {str(e)}")
                continue
    
    return params_dict

def main():
    """Main execution function with command line argument support"""
    parser = argparse.ArgumentParser(description='Audio Classification and Matching Tool')
    parser.add_argument('--classify', action='store_true', 
                       help='Run audio classification on WAV files')
    parser.add_argument('--match', type=str, 
                       help='Find best match for given prompt (e.g., "ambient strings")')
    parser.add_argument('--audio-dir', type=str, 
                       default=r"C:\Users\stuart\Documents\python_projects\ouput_midi_input_audo",
                       help='Directory containing audio files')
    parser.add_argument('--output-csv', type=str,
                       default=r"C:\Users\stuart\Documents\python_projects\ouput_midi_input_audo\audio_analysis.csv",
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Set environment variables before any imports
    os.environ["OPENAI_LOG"] = "warning"
    os.environ["HTTPX_LOG_LEVEL"] = "WARNING"
    
    # Configure based on arguments
    config = Config()
    if args.audio_dir:
        config.audio_dir = args.audio_dir
    if args.output_csv:
        config.output_csv = args.output_csv
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
        ]
    )
    
    # Suppress specific loggers
    for logger_name in ["openai", "httpx", "urllib3", "requests", "httpcore"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    try:
        if args.classify:
            # Run audio classification
            print("Starting audio classification...")
            processor = AudioProcessor(config)
            processor.process_audio_files()
            print("✓ Audio classification complete!")
            
        elif args.match:
            # Run matching functionality
            print(f"Finding matches for: {args.match}")
            result = parse_user_prompt(args.match)
            best_match_index = find_best_matches_with_llm(result, config.output_csv)
            
            if best_match_index:
                print(f"\n✓ Best match found: Index {best_match_index}")
            else:
                print("\n✗ No suitable match found")
                
        else:
            # Default behavior - run matching with default prompt
            print("No arguments provided. Running default match search...")
            user_prompt = "ambient strings"
            result = parse_user_prompt(user_prompt)
            best_match_index = find_best_matches_with_llm(result, config.output_csv)
            
            if best_match_index:
                print(f"\n✓ Best match found: Index {best_match_index}")
            else:
                print("\n✗ No suitable match found")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
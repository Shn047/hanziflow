// --- CONFIGURATION ---
const FASTAPI_URL = 'http://192.168.1.173:8000/api/v1/segment'; 
const ANALYZE_URL = 'http://192.168.1.173:8000/api/v1/analyze';
const INTERNAL_API_KEY = YOUR_API_KEY; 

console.log("--- HanziFlow: background.js SCRIPT RUNNING ---");

// --- MESSAGE LISTENER ---
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("--- HanziFlow: background.js received message:", request.action);
    if (request.action === "segmentChinese") {
        handleSegmentationRequest(request.text, sendResponse);
        return true; 
    }
});

// --- REQUEST HANDLER ---
async function handleSegmentationRequest(text, sendResponse) {
    console.log("--- HanziFlow: Text received for API call:", `"${text}"`);
    console.log("--- HanziFlow: background.js: Sending request to local backend...");

    try {
        // 1) Segmentation is required
        const segmentResponse = await callRealAIBackend(text);

        // 2) Usage/grammar is optional – failure here should not kill the whole flow
        let mergedResponse = { ...segmentResponse };
        try {
            const analyzeResponse = await callUsageBackend(text);
            const usagePatterns = mapReportToUsagePatterns(analyzeResponse.report, text);
            mergedResponse.usage_patterns = usagePatterns;
        } catch (usageError) {
            console.warn("--- HanziFlow: background.js: analyze call failed, continuing without usage patterns:", usageError.message);
        }

        // 3) Parse combined data into the frontend shape
        const formattedData = parseModelSegmentation(text, mergedResponse);

        console.log("--- HanziFlow: background.js: Successfully parsed model + usage output");
        sendResponse({ success: true, data: formattedData });

    } catch (error) {
        console.error("--- HanziFlow: background.js: CATCH BLOCK ERROR ---", error.message);
        sendResponse({ success: false, error: error.message });
    }
}

// --- API CALL FUNCTION ---
async function callRealAIBackend(chineseText) {
    const response = await fetch(FASTAPI_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': INTERNAL_API_KEY 
        },
        body: JSON.stringify({
            content: chineseText 
        })
    });

    if (!response.ok) {
        let errorDetail = `API error: ${response.status}`;
        try {
            const errorData = await response.json();
            errorDetail = errorData.detail || errorDetail; 
        } catch (e) {
            errorDetail = `API error: ${response.status} ${response.statusText}`;
        }
        console.error("--- HanziFlow: background.js: API response not OK:", errorDetail);
        throw new Error(errorDetail);
    }

    return await response.json(); 
}

// --- USAGE / GRAMMAR API CALL FUNCTION ---
async function callUsageBackend(chineseText) {
    const response = await fetch(ANALYZE_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': INTERNAL_API_KEY
        },
        body: JSON.stringify({
            content: chineseText
        })
    });

    if (!response.ok) {
        let errorDetail = `API error: ${response.status}`;
        try {
            const errorData = await response.json();
            errorDetail = errorData.detail || errorDetail;
        } catch (e) {
            errorDetail = `API error: ${response.status} ${response.statusText}`;
        }
        console.error("--- HanziFlow: background.js: ANALYZE API not OK:", errorDetail);
        throw new Error(errorDetail);
    }

    return await response.json(); 
}

// --- MAP ANALYZE REPORT TO USAGE PATTERNS ---
function mapReportToUsagePatterns(reportText, originalText) {
    if (!reportText || typeof reportText !== 'string') {
        return [];
    }
    let explanation = reportText.trim();
    const marker = 'GRAMMAR PATTERN';
    const idx = reportText.indexOf(marker);
    if (idx !== -1) {
        explanation = reportText.slice(idx + marker.length).trim();
    }

    return [
        {
            phrase: originalText,
            explanation: explanation,
            example: ''  
        }
    ];
}

// --- SEGMENTATION PARSER ---
function parseModelSegmentation(originalText, modelResponse) {
    console.log("--- HanziFlow: Parsing model response:", modelResponse);

    let segmentsString = '';

    if (modelResponse && typeof modelResponse === 'object') {
        if (typeof modelResponse.segments === 'string') {
            segmentsString = modelResponse.segments;
        } else if (typeof modelResponse.segmented_sentence === 'string') {
            segmentsString = modelResponse.segmented_sentence;
        } else if (typeof modelResponse.result === 'string') {
            segmentsString = modelResponse.result;
        }
    } else if (typeof modelResponse === 'string') {
        segmentsString = modelResponse;
    }

    if (!segmentsString || !segmentsString.trim()) {
        console.warn("--- HanziFlow: No usable segmentsString, falling back to character segmentation.");
        segmentsString = originalText.split('').join(' ');
    }

    const segments = cleanAndParseSegments(segmentsString);

    let words = [];
    if (modelResponse && Array.isArray(modelResponse.words) && modelResponse.words.length > 0) {
        words = modelResponse.words;
    } else {
        words = createWordObjects(segments);
    }

    let usagePatterns = [];
    if (modelResponse && Array.isArray(modelResponse.usage_patterns)) {
        usagePatterns = modelResponse.usage_patterns;
    } else {
        usagePatterns = [];
    }

    return {
        segmented_sentence: segmentsString,
        words,
        usage_patterns: usagePatterns
    };
}

// --- SEGMENT CLEANING AND PARSING ---
function cleanAndParseSegments(segmentsString) {
    console.log("--- HanziFlow: Raw segments string:", segmentsString);
    
    let cleaned = segmentsString
        .replace(/\s+/g, ' ')          
        .replace(/^\\s+|\\s+$/g, '')   
        .trim();
    
    console.log("--- HanziFlow: Cleaned segments string:", cleaned);
    
    // Split by spaces to get individual segments
    const segments = cleaned.split(' ').filter(segment => {
        return segment.length > 0 && segment !== ' ' && segment !== '\\s';
    });
    
    console.log("--- HanziFlow: Parsed segments array:", segments);
    return segments;
}

// --- CREATE WORD OBJECTS FOR FRONTEND ---
function createWordObjects(segments) {
    return segments.map(segment => {
        const translation = getTranslationForWord(segment);
        const pos = guessPartOfSpeech(segment);
        
        return {
            word: segment,
            translation: translation,
            pos: pos,
            pinyin: '' 
        };
    });
}

// --- SIMPLE TRANSLATION LOOKUP (ENHANCE LATER) ---
function getTranslationForWord(chineseWord) {
    const translationMap = {
        '的': "'s/of",
        '是': 'is/are',
        '在': 'at/in',
        '有': 'have/has',
        '和': 'and',
        '了': '(past tense marker)',
        '我': 'I/me',
        '你': 'you',
        '他': 'he/him',
        '她': 'she/her',
        '它': 'it',
        '我们': 'we/us',
        '他们': 'they/them',
        '这个': 'this',
        '那个': 'that',
        '什么': 'what',
        '为什么': 'why',
        '怎么': 'how',
        '哪里': 'where',
        '谁': 'who',
        '时候': 'time/when',
        '今天': 'today',
        '明天': 'tomorrow',
        '昨天': 'yesterday',
        '天气': 'weather',
        '很好': 'very good',
        '喜欢': 'like',
        '爱': 'love',
        '看': 'look/see/watch',
        '听': 'listen/hear',
        '说': 'speak/say',
        '吃': 'eat',
        '喝': 'drink',
        '去': 'go',
        '来': 'come',
        '做': 'do/make',
        '工作': 'work',
        '学习': 'study/learn',
        '学校': 'school',
        '老师': 'teacher',
        '学生': 'student',
        '朋友': 'friend',
        '家庭': 'family',
        '爸爸': 'father',
        '妈妈': 'mother',
        '哥哥': 'older brother',
        '弟弟': 'younger brother',
        '姐姐': 'older sister',
        '妹妹': 'younger sister'
    };
    
    return translationMap[chineseWord] || chineseWord;
}

// --- SIMPLE PART-OF-SPEECH GUESSING ---
function guessPartOfSpeech(word) {
    if (word.length === 1 && ['的', '了', '着', '过', '啊', '呢', '吗', '吧'].includes(word)) {
        return 'particle';
    }
    if (['我', '你', '他', '她', '它', '我们', '你们', '他们'].includes(word)) {
        return 'pronoun';
    }
    if (['是', '有', '在', '做', '看', '听', '说', '吃', '喝', '去', '来'].includes(word)) {
        return 'verb';
    }
    if (['很', '非常', '太', '更', '最'].includes(word)) {
        return 'adverb';
    }
    if (['好', '大', '小', '多', '少', '新', '旧', '红', '绿', '蓝'].includes(word)) {
        return 'adjective';
    }
    
    return 'noun'; // Default to noun
}

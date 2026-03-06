const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { OpenAI } = require('openai');
const pdf = require('pdf-parse');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { OpenAIEmbeddings } = require('@langchain/openai');
require('dotenv').config();

const app = express();
app.use(cors({ origin: '*' }));
app.use(express.json());

// אתחול OpenAI
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const embeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });

let vectorStore = null;

// פונקציה לטעינה ועיבוד של קבצי ה-PDF מהתיקייה docs
async function initializeRAG() {
    try {
        console.log("--- Starting PDF processing ---");
        const docsPath = path.join(__dirname, 'docs');
        
        if (!fs.existsSync(docsPath)) {
            console.log("Docs folder not found. Creating one...");
            fs.mkdirSync(docsPath);
            return;
        }

        const files = fs.readdirSync(docsPath).filter(f => f.endsWith('.pdf'));
        console.log(`Found ${files.length} PDF files:`, files);

        if (files.length === 0) {
            console.log("No PDF files found to process.");
            return;
        }

        let combinedText = "";
        for (const file of files) {
            try {
                const dataBuffer = fs.readFileSync(path.join(docsPath, file));
                const data = await pdf(dataBuffer);
                combinedText += data.text + "\n\n";
                console.log(`Successfully parsed: ${file}`);
            } catch (err) {
                console.error(`Error parsing ${file}:`, err.message);
            }
        }

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const docs = await splitter.createDocuments([combinedText]);

        console.log("Generating embeddings and building vector store...");
        vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
        console.log("--- RAG system initialized and ready! 📚 ---");
    } catch (error) {
        console.error("CRITICAL ERROR during initialization:", error);
    }
}

const SYSTEM_PROMPT = `אתה יועץ משפטי מומחה בתחום חדלות פרעון בישראל.
ענה תמיד בעברית בלבד. התבסס על המידע המקצועי מהמסמכים שסופקו.
אם המידע לא קיים במסמכים, ענה מידע כללי וציין זאת.
תמיד סיים בהבהרה שהמידע אינו מהווה ייעוץ משפטי פרטני.`;

app.post('/api/chat', async (req, res) => {
    try {
        const { question, history } = req.body;
        if (!question) return res.status(400).json({ error: "No question" });

        let context = "";
        if (vectorStore) {
            const searchResults = await vectorStore.similaritySearch(question, 3);
            context = searchResults.map(r => r.pageContent).join("\n---\n");
        }

        const formattedHistory = (history || []).map(msg => ({
            role: msg.role === 'bot' ? 'assistant' : 'user',
            content: msg.content
        }));

        const response = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "system", content: `מידע מהמסמכים:\n${context}` },
                ...formattedHistory,
                { role: "user", content: question }
            ],
        });

        res.json({ reply: response.choices[0].message.content });
    } catch (error) {
        console.error("API Error:", error);
        res.status(500).json({ error: "Internal Error" });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, async () => {
    console.log(`Server running on port ${PORT}`);
    await initializeRAG();
});

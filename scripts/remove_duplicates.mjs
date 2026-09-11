import fs from 'fs';
const data = JSON.parse(fs.readFileSync('src/data/models.json', 'utf8'));
const cleaned = data.filter(m => !m.name.includes("gpt oss") && !m.name.includes("0731"));
fs.writeFileSync('src/data/models.json', JSON.stringify(cleaned, null, 2));

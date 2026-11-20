import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from 'file:///C:/Program%20Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1110-973-0_26.1110.973.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const deck = await PresentationFile.importPptx(await FileBlob.load('C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Natural Language Processing/AI_Paraphraser/.level-sensor-build/candidate.pptx'));
for (let i=0; i<deck.slides.items.length; i++) {
  const png = await deck.export({slide: deck.slides.items[i], format:'png', scale:1});
  await fs.writeFile(`C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Natural Language Processing/AI_Paraphraser/.level-sensor-build/render-${String(i+1).padStart(2,'0')}.png`, new Uint8Array(await png.arrayBuffer()));
}
console.log(deck.slides.items.length);

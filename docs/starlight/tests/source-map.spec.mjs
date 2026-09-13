import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';

test('prose selections retain exact origins in authored and included files', async ({page})=>{
  for(const [route,file] of [['/','docs/source/index.md'],['/developer/','CONTRIBUTING.md'],['/CHANGELOG/','CHANGELOG.md']]) {
    await page.goto('.' + route);
    const result=await page.evaluate(file=>{
      const span=[...document.querySelectorAll('main [data-source-location]')].find(el=>{
        const origin=JSON.parse(el.dataset.sourceLocation);
        return origin.file===file && origin.kind==='exact' && el.textContent.length>30;
      });
      // Selections can cross inline branding and other nested markup.
      const walker=document.createTreeWalker(span,NodeFilter.SHOW_TEXT);
      const range=document.createRange();
      let offset=0;
      while(walker.nextNode()) {
        const node=walker.currentNode;
        const end=offset+node.textContent.length;
        if(offset<=2 && end>2) range.setStart(node,2-offset);
        if(offset<20 && end>=20) {range.setEnd(node,20-offset);break;}
        offset=end;
      }
      return window.mystSourceMap.resolve(range);
    },file);
    expect(result.complete).toBe(true);
    expect(result.ranges).toHaveLength(1);
    const {origin}=result.ranges[0];
    expect(origin.file).toBe(file);
    const source=await readFile(resolve('../..',file),'utf8');
    expect(source.slice(origin.start,origin.end)).toBe(result.text);
  }
});

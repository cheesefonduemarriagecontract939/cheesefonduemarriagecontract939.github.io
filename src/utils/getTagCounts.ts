import type { CollectionEntry } from "astro:content";
import { postFilter } from "./postFilter";
import { slugifyStr } from "./slugify";

type TagStat = {
  tag: string;
  tagName: string;
  count: number;
};

export function getTagCounts(posts: CollectionEntry<"posts">[]): TagStat[] {
  const tagMap = new Map<string, TagStat>();

  for (const post of posts.filter(postFilter)) {
    for (const tagName of post.data.tags) {
      const tag = slugifyStr(tagName);
      const current = tagMap.get(tag);

      if (current) {
        current.count += 1;
      } else {
        tagMap.set(tag, { tag, tagName, count: 1 });
      }
    }
  }

  return [...tagMap.values()].sort(
    (a, b) => b.count - a.count || a.tagName.localeCompare(b.tagName, "zh-CN")
  );
}

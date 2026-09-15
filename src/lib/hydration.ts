// Module-level singleton tracking whether the store has been hydrated from
// the URL. TopNav's URL writer must not run before this flips true, or it
// clobbers shared links with default state before hydration completes.
let hydrated = false;

export function markHydrated(): void {
  hydrated = true;
}

export function isHydrated(): boolean {
  return hydrated;
}

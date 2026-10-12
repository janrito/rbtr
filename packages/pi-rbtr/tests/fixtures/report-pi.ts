/**
 * An extension that writes the package directory and version of the pi
 * loading it to $PI_REPORT, as JSON. `vitest.config.ts` loads it into the
 * installed `pi` to find the pi the tests run against.
 */

import { writeFileSync } from "node:fs";
import { getPackageDir, VERSION } from "@earendil-works/pi-coding-agent";

export default function reportPi(): void {
  writeFileSync(process.env.PI_REPORT ?? "", JSON.stringify({ dir: getPackageDir(), version: VERSION }));
}

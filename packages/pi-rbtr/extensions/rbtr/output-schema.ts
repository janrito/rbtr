/**
 * Output schemas for the rbtr tools, built from rbtr's protocol schema.
 */

import { type TSchema, Type } from "typebox";

import { PROTOCOL_DEFS } from "./generated/schemas.js";

type ProtocolDef = keyof typeof PROTOCOL_DEFS;

/** The JSON Schema of a reply that is any one of the named protocol definitions. */
export function replySchema(...names: [ProtocolDef, ...ProtocolDef[]]): TSchema {
  return Type.Unsafe({ $defs: PROTOCOL_DEFS, anyOf: names.map((name) => ({ $ref: `#/$defs/${name}` })) });
}

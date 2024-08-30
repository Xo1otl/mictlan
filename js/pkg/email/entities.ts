export class ValueObject {
  constructor(private value: string) {}
  toString(): string {
    return this.value;
  }
}
